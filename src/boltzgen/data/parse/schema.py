from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import replace, astuple
from collections import defaultdict
from pathlib import Path
import random
import re
from typing import Optional
from copy import deepcopy

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Conformer, Mol
from rdkit.Chem.rdMolDescriptors import CalcNumHeavyAtoms
from scipy.spatial.distance import cdist
import yaml

from boltzgen.data import const
from boltzgen.data.mol import load_molecules
from boltzgen.data.parse.mmcif import parse_mmcif
from boltzgen.data.data import (
    Atom,
    Bond,
    Chain,
    ChainInfo,
    Coords,
    DesignInfo,
    Ensemble,
    Interface,
    Record,
    Residue,
    Structure,
    StructureInfo,
    Target,
    Token,
    Tokenized,
)
from boltzgen.data.parse.pdb_parser import parse_pdb
from boltzgen.data.tokenize.tokenizer import TokenData
from dataclasses import replace

####################################################################################################
# DATACLASSES
####################################################################################################


@dataclass(frozen=True)
class ParsedAtom:
    """A parsed atom object."""

    name: str
    element: int
    charge: int
    coords: tuple[float, float, float]
    conformer: tuple[float, float, float]
    is_present: bool
    chirality: int


@dataclass(frozen=True)
class ParsedBond:
    """A parsed bond object."""

    atom_1: int
    atom_2: int
    type: int


@dataclass(frozen=True)
class ParsedRDKitBoundsConstraint:
    """A parsed RDKit bounds constraint object."""

    atom_idxs: tuple[int, int]
    is_bond: bool
    is_angle: bool
    upper_bound: float
    lower_bound: float


@dataclass(frozen=True)
class ParsedChiralAtomConstraint:
    """A parsed chiral atom constraint object."""

    atom_idxs: tuple[int, int, int, int]
    is_reference: bool
    is_r: bool


@dataclass(frozen=True)
class ParsedStereoBondConstraint:
    """A parsed stereo bond constraint object."""

    atom_idxs: tuple[int, int, int, int]
    is_check: bool
    is_e: bool


@dataclass(frozen=True)
class ParsedPlanarBondConstraint:
    """A parsed planar bond constraint object."""

    atom_idxs: tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class ParsedPlanarRing5Constraint:
    """A parsed planar bond constraint object."""

    atom_idxs: tuple[int, int, int, int, int]


@dataclass(frozen=True)
class ParsedPlanarRing6Constraint:
    """A parsed planar bond constraint object."""

    atom_idxs: tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class ParsedResidue:
    """A parsed residue object."""

    name: str
    type: int
    idx: int
    atoms: list[ParsedAtom]
    bonds: list[ParsedBond]
    orig_idx: Optional[int]
    atom_center: int
    atom_disto: int
    is_standard: bool
    is_present: bool


@dataclass(frozen=True)
class ParsedChain:
    """A parsed chain object."""

    entity: str
    type: int
    residues: list[ParsedResidue]
    res_design_mask: list[bool]
    cyclic_period: int
    sequence: Optional[str] = None
    sampleidx_to_specidx: Optional[np.ndarray] = None


@dataclass(frozen=True)
class Alignment:
    """A parsed alignment object."""

    query_st: int
    query_en: int
    template_st: int
    template_en: int


####################################################################################################
# HELPERS
####################################################################################################


def compute_3d_conformer(mol: Mol, version: str = "v3") -> bool:
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
        options = AllChem.ETKDGv3()
    elif version == "v2":
        options = AllChem.ETKDGv2()
    else:
        options = AllChem.ETKDGv2()

    options.clearConfs = False
    conf_id = -1

    try:
        conf_id = AllChem.EmbedMolecule(mol, options)

        if conf_id == -1:
            print(
                f"WARNING: RDKit ETKDGv3 failed to generate a conformer for molecule "
                f"{Chem.MolToSmiles(AllChem.RemoveHs(mol))}, so the program will start with random coordinates. "
                f"Note that the performance of the model under this behaviour was not tested."
            )
            options.useRandomCoords = True
            conf_id = AllChem.EmbedMolecule(mol, options)

        AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=1000)

    except RuntimeError:
        pass  # Force field issue here
    except ValueError:
        pass  # sanitization issue here

    if conf_id != -1:
        conformer = mol.GetConformer(conf_id)
        conformer.SetProp("name", "Computed")
        conformer.SetProp("coord_generation", f"ETKDG{version}")

        return True

    return False


def get_conformer(mol: Mol) -> Conformer:
    """Retrieve an rdkit object for a deemed conformer.

    Inspired by `pdbeccdutils.core.component.Component`.

    Parameters
    ----------
    mol: Mol
        The molecule to process.

    Returns
    -------
    Conformer
        The desired conformer, if any.

    Raises
    ------
    ValueError
        If there are no conformers of the given tyoe.

    """
    # Try using the computed conformer
    for c in mol.GetConformers():
        try:
            if c.GetProp("name") == "Computed":
                return c
        except KeyError:  # noqa: PERF203
            pass

    # Fallback to the ideal coordinates
    for c in mol.GetConformers():
        try:
            if c.GetProp("name") == "Ideal":
                return c
        except KeyError:  # noqa: PERF203
            pass

    # Fallback to boltz2 format
    conf_ids = [int(conf.GetId()) for conf in mol.GetConformers()]
    if len(conf_ids) > 0:
        conf_id = conf_ids[0]
        conformer = mol.GetConformer(conf_id)
        return conformer

    msg = "Conformer does not exist."
    raise ValueError(msg)


def get_mol(ccd: str, mols: dict, moldir: str) -> Mol:
    """Get mol from CCD code.

    Return mol with ccd from mols if it is in mols. Otherwise load it from moldir,
    add it to mols, and return the mol.
    """
    mol = mols.get(ccd)

    if mol is None:
        mol = load_molecules(moldir, [ccd])[ccd]
        mols[ccd] = mol  # cache for future calls

    return mol


####################################################################################################
# PARSING
####################################################################################################

yaml_keys = [
    "entities",
    "protein",
    "dna",
    "rna",
    "id",
    "sequence",
    "ligand",
    "ccd",
    "secondary_structure",
    "file",
    "path",
    "msa",
    "include",
    "chain",
    "include_proximity",
    "res_index",
    "radius",
    "binding_types",
    "binding",
    "not_binding",
    "structure_groups",
    "group",
    "visibility",
    "design",
    "loop",
    "helix",
    "sheet",
    "design_insertions",
    "insertion",
    "num_residues",
    "fuse",
    "exclude",
    "smiles",
    "cyclic",
    "bonds",
    "bond",
    "atom1",
    "atom2",
    "bondtype",
    "structure_group",
    "constraints",
    "total_len",
    "min",
    "max",
    "reset_res_index",
    "not_design",
    "leaving_atoms",
    "atom",
    "use_assembly",
]


def parse_ccd_residue(
    name: str,
    ref_mol: Mol,
    res_idx: int,
) -> Optional[ParsedResidue]:
    """Parse an MMCIF ligand.

    First tries to get the SMILES string from the RCSB.
    Then, tries to infer atom ordering using RDKit.

    Parameters
    ----------
    name: str
        The name of the molecule to parse.
    ref_mol: Mol
        The reference molecule to parse.
    res_idx : int
        The residue index.

    Returns
    -------
    ParsedResidue, optional
       The output ParsedResidue, if successful.

    """
    unk_chirality = const.chirality_type_ids[const.unk_chirality_type]

    # Check if this is a single heavy atom CCD residue
    if CalcNumHeavyAtoms(ref_mol) == 1:
        # Remove hydrogens
        ref_mol = AllChem.RemoveHs(ref_mol, sanitize=False)

        pos = (0, 0, 0)
        ref_atom = ref_mol.GetAtoms()[0]
        chirality_type = const.chirality_type_ids.get(
            str(ref_atom.GetChiralTag()), unk_chirality
        )
        atom = ParsedAtom(
            name=ref_atom.GetProp("name"),
            element=ref_atom.GetAtomicNum(),
            charge=ref_atom.GetFormalCharge(),
            coords=pos,
            conformer=(0, 0, 0),
            is_present=True,
            chirality=chirality_type,
        )
        unk_prot_id = const.unk_token_ids["PROTEIN"]
        residue = ParsedResidue(
            name=name,
            type=unk_prot_id,
            atoms=[atom],
            bonds=[],
            idx=res_idx,
            orig_idx=None,
            atom_center=0,  # Placeholder, no center
            atom_disto=0,  # Placeholder, no center
            is_standard=False,
            is_present=True,
        )
        return residue

    # Get reference conformer coordinates
    conformer = get_conformer(ref_mol)

    # Parse each atom in order of the reference mol
    atoms = []
    atom_idx = 0
    idx_map = {}  # Used for bonds later

    for i, atom in enumerate(ref_mol.GetAtoms()):
        # Ignore Hydrogen atoms
        if atom.GetAtomicNum() == 1:
            continue

        # Get atom name, charge, element and reference coordinates
        atom_name = atom.GetProp("name")
        charge = atom.GetFormalCharge()
        element = atom.GetAtomicNum()
        ref_coords = conformer.GetAtomPosition(atom.GetIdx())
        ref_coords = (ref_coords.x, ref_coords.y, ref_coords.z)
        chirality_type = const.chirality_type_ids.get(
            str(atom.GetChiralTag()), unk_chirality
        )

        # Get PDB coordinates, if any
        coords = (0, 0, 0)
        atom_is_present = True

        # Add atom to list
        atoms.append(
            ParsedAtom(
                name=atom_name,
                element=element,
                charge=charge,
                coords=coords,
                conformer=ref_coords,
                is_present=atom_is_present,
                chirality=chirality_type,
            )
        )
        idx_map[i] = atom_idx
        atom_idx += 1  # noqa: SIM113

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
        orig_idx=None,
        is_standard=False,
        is_present=True,
    )


def parse_polymer(
    sequence: list[str],
    raw_sequence: str,
    entity: str,
    chain_type: str,
    components: dict[str, Mol],
    cyclic: bool,
    mol_dir: Path,
) -> Optional[ParsedChain]:
    """Process a sequence into a chain object.

    Performs alignment of the full sequence to the polymer
    residues. Loads coordinates and masks for the atoms in
    the polymer, following the ordering in const.atom_order.

    Parameters
    ----------
    sequence : list[str]
        The full sequence of the polymer.
    entity : str
        The entity id.
    entity_type : str
        The entity type.
    components : dict[str, Mol]
        The preprocessed PDB components dictionary.

    Returns
    -------
    ParsedChain, optional
        The output chain, if successful.

    Raises
    ------
    ValueError
        If the alignment fails.

    """
    ref_res = set(const.tokens)
    unk_chirality = const.chirality_type_ids[const.unk_chirality_type]

    # Make sequence and distinguish between design and non-design
    seq_processed = []
    res_design_mask = []
    sampleidx_to_specidx = []
    count = 0
    for token in sequence:
        if isinstance(token, str):
            seq_processed.append(token)
            res_design_mask.append(False)
            sampleidx_to_specidx.append(count)
            count += 1
        elif isinstance(token, tuple):
            if len(token) == 1:
                num = start = token[0]
                sampleidx_to_specidx.extend(range(count, count + num))
            elif len(token) == 2:
                start, end = token
                num = np.random.randint(start, end + 1)
                mapping = list(range(count, count + start))
                mapping += [count + start - 1] * (num - start)
                sampleidx_to_specidx.extend(mapping)
            res_design_mask.extend([True] * num)
            seq_processed.extend(["GLY"] * num)
            count += start
        else:
            raise ValueError("Token must be tuple of int or string")
    sampleidx_to_specidx = np.array(sampleidx_to_specidx)

    # Get coordinates and masks
    parsed = []
    for res_idx, res_name in enumerate(seq_processed):
        # Check if modified residue
        # Map MSE to MET
        res_corrected = res_name if res_name != "MSE" else "MET"

        # Handle non-standard residues
        if res_corrected not in ref_res:
            ref_mol = get_mol(res_corrected, components, mol_dir)
            residue = parse_ccd_residue(
                name=res_corrected,
                ref_mol=ref_mol,
                res_idx=res_idx,
            )
            parsed.append(residue)
            continue

        # Load ref residue
        ref_mol = get_mol(res_corrected, components, mol_dir)
        ref_mol = AllChem.RemoveHs(ref_mol, sanitize=False)
        ref_conformer = get_conformer(ref_mol)

        # Only use reference atoms set in constants
        ref_name_to_atom = {a.GetProp("name"): a for a in ref_mol.GetAtoms()}
        ref_atoms = [ref_name_to_atom[a] for a in const.ref_atoms[res_corrected]]

        # Iterate, always in the same order
        atoms: list[ParsedAtom] = []

        for ref_atom in ref_atoms:
            # Get atom name
            atom_name = ref_atom.GetProp("name")
            idx = ref_atom.GetIdx()

            # Get conformer coordinates
            ref_coords = ref_conformer.GetAtomPosition(idx)
            ref_coords = (ref_coords.x, ref_coords.y, ref_coords.z)

            # Set 0 coordinate
            atom_is_present = True
            coords = (0, 0, 0)

            # Add atom to list
            atoms.append(
                ParsedAtom(
                    name=atom_name,
                    element=ref_atom.GetAtomicNum(),
                    charge=ref_atom.GetFormalCharge(),
                    coords=coords,
                    conformer=ref_coords,
                    is_present=atom_is_present,
                    chirality=const.chirality_type_ids.get(
                        str(ref_atom.GetChiralTag()), unk_chirality
                    ),
                )
            )

        atom_center = const.res_to_center_atom_id[res_corrected]
        atom_disto = const.res_to_disto_atom_id[res_corrected]
        parsed.append(
            ParsedResidue(
                name=res_corrected,
                type=const.token_ids[res_corrected],
                atoms=atoms,
                bonds=[],
                idx=res_idx,
                atom_center=atom_center,
                atom_disto=atom_disto,
                is_standard=True,
                is_present=True,
                orig_idx=None,
            )
        )
    if cyclic:
        cyclic_period = len(seq_processed)
    else:
        cyclic_period = 0
    # Return polymer object
    return ParsedChain(
        entity=entity,
        residues=parsed,
        res_design_mask=res_design_mask,
        type=chain_type,
        cyclic_period=cyclic_period,
        sequence=raw_sequence,
        sampleidx_to_specidx=sampleidx_to_specidx,
    )


# Define helper
def parse_range(ranges, c_start=0, c_end=None):
    ranges = str(ranges)
    if "," in ranges:
        spec_list = ranges.split(",")
    else:
        spec_list = [ranges]

    indices = []
    for spec in spec_list:
        if re.fullmatch(r"\d+", spec):
            # Single number. Convert it from 1 indexed to 0 indexed.
            start = int(spec) - 1
            end = int(spec) - 1
            indices.append(c_start + start)
        elif re.fullmatch(r"\d+..\d+", spec):
            # Range with start and end. Convert the start from 1 indexed to 0 indexed. Leave the end untouched because the specification is inclusive (+1) but 1 indexed (-1).
            start, end = map(int, spec.split(".."))
            start -= 1
            indices += list(range(c_start + start, c_start + end))
        elif re.fullmatch(r"..\d+", spec):
            # Range that is inclusive of the specified end (which is specified in a 1 indexed fashion).
            end = int(spec.replace("..", ""))
            start = 0
            indices += list(range(c_start, c_start + end))
        elif re.fullmatch(r"\d+..", spec):
            assert c_end is not None
            # Range that is inclusive of the specified start (which is specified in a 1 indexed fashion).
            start = int(spec.replace("..", ""))
            start -= 1
            end = c_end - c_start
            indices += list(range(c_start + start, c_end))
    if start < 0:
        msg = f"There is a 0 in the specified range(s) {ranges}. Residue indices are 1 indexed."
        raise ValueError(msg)

    if c_end is not None and end > c_end - c_start:
        msg = f"Specified end {ranges} is higher than the length of the chain."
        raise ValueError(msg)
    return indices


def parse_entity(item, mols, mol_dir, ligand_id, is_msa_custom, is_msa_auto):
    extra_mols: dict[str, Mol] = {}
    parsed_chains: dict[str, ParsedChain] = {}
    res_bind_type: list[int] = []
    ss_type: list[int] = []
    chain_to_msa: dict[str, str] = {}
    # Get entity type and sequence
    entity_type = next(iter(item.keys())).lower()

    # Ensure all the items share the same msa
    msa = -1
    if entity_type == "protein":
        designed = bool(re.search(r"\d", str(item[entity_type]["sequence"])))
        if designed:
            # Get the msa, default to -1, meaning no msa.
            msa = item[entity_type].get("msa", -1)
            if (msa is None) or (msa == ""):
                msa = -1
        else:
            # Get the msa, default to 0, meaning auto-generated
            msa = item[entity_type].get("msa", 0)
            if (msa is None) or (msa == ""):
                msa = 0

        # Check if all MSAs are the same within the same entity
        item_msa = item[entity_type].get("msa", 0)
        if (item_msa is None) or (item_msa == ""):
            item_msa = 0

        if item_msa != msa and not designed:
            msg = "All proteins with the same sequence must share the same MSA!"
            raise ValueError(msg)

        # Set the MSA, warn if passed in single-sequence mode
        if msa == "empty":
            msa = -1
            msg = (
                "Found explicit empty MSA for some proteins, will run "
                "these in single sequence mode."
            )
            print(msg)

        if msa not in (0, -1):
            is_msa_custom = True
        elif msa == 0:
            is_msa_auto = True

    # Parse a polymer
    if entity_type in {"protein", "dna", "rna"}:
        # Get token map
        if entity_type == "rna":
            token_map = const.rna_letter_to_token
        elif entity_type == "dna":
            token_map = const.dna_letter_to_token
        elif entity_type == "protein":
            token_map = const.prot_letter_to_token
        else:
            msg = f"Unknown polymer type: {entity_type}"
            raise ValueError(msg)

        # Get polymer info
        chain_type = const.chain_type_ids[entity_type.upper()]
        unk_token = const.unk_token[entity_type.upper()]

        # Extract sequence
        raw_seq = str(item[entity_type]["sequence"])

        # Convert sequence to standard and design tokens
        seq = []
        parts = re.split(r",\s*", raw_seq)  # split by comma (optional whitespace after)
        for part in parts:
            # If a part is empty (e.g., from "1,,2"), skip it.
            if not part:
                continue
            tokens = re.findall(r"\d+\.\.\d+|\d+|[a-zA-Z]", part)
            for token in tokens:
                if re.fullmatch(r"\d+\.\.\d+", token):  # Case 2: range
                    start, end = map(int, token.split(".."))
                    seq.append((start, end))
                elif re.fullmatch(r"\d+", token):  # Case 1: single number
                    seq.append((int(token),))
                else:  # Case 3: characters
                    seq.extend([token_map.get(c, unk_token) for c in token])

        # Apply modifications
        for mod in item[entity_type].get("modifications", []):
            code = mod["ccd"].upper()
            idx = mod["position"] - 1  # 1-indexed
            seq[idx] = code

        cyclic = item[entity_type].get("cyclic", False)

        # Parse a polymer
        parsed_chain = parse_polymer(
            sequence=seq,
            raw_sequence=raw_seq,
            entity=0,
            chain_type=chain_type,
            components=mols,
            cyclic=cyclic,
            mol_dir=mol_dir,
        )

    # Parse a non-polymer
    elif (entity_type == "ligand") and "ccd" in (item[entity_type]):
        seq = item[entity_type]["ccd"]
        if isinstance(seq, str):
            seq = [seq]

        residues = []
        for res_idx, code in enumerate(seq):
            code = code.upper()
            # Get mol
            ref_mol = get_mol(code, mols, mol_dir)

            # Parse residue
            residue = parse_ccd_residue(
                name=code,
                ref_mol=ref_mol,
                res_idx=res_idx,
            )
            residues.append(residue)

        # Create multi ligand chain
        parsed_chain = ParsedChain(
            entity=0,
            residues=residues,
            res_design_mask=[False] * len(residues),
            type=const.chain_type_ids["NONPOLYMER"],
            cyclic_period=0,
            sequence=None,
        )

        assert not item[entity_type].get("cyclic", False), (
            "Cyclic flag is not supported for ligands"
        )

    elif (entity_type == "ligand") and ("smiles" in item[entity_type]):
        seq = item[entity_type]["smiles"]
        mol = AllChem.MolFromSmiles(seq)
        mol = AllChem.AddHs(mol)
        element_counts = defaultdict(int)
        for i, atom in enumerate(mol.GetAtoms()):
            symbol = atom.GetSymbol()
            element_counts[symbol] += 1
            atom_name = f"{symbol}{element_counts[symbol]}"
            if len(atom_name) > 4:
                raise ValueError(
                    f"{seq} has an atom with a name longer than 4 characters: {atom_name}"
                )
            atom.SetProp("name", atom_name)
        success = compute_3d_conformer(mol)

        if not success:
            msg = f"Failed to compute 3D conformer for {seq}"
            raise ValueError(msg)

        mol_no_h = AllChem.RemoveHs(mol)
        extra_mols[f"LIG{ligand_id}"] = mol_no_h

        residue = parse_ccd_residue(
            name=f"LIG{ligand_id}",
            ref_mol=mol,
            res_idx=0,
        )
        ligand_id += 1
        parsed_chain = ParsedChain(
            entity=0,
            residues=[residue],
            res_design_mask=[False],
            type=const.chain_type_ids["NONPOLYMER"],
            cyclic_period=0,
            sequence=None,
        )

        assert not item[entity_type].get("cyclic", False), (
            "Cyclic flag is not supported for ligands"
        )
    elif entity_type == "file":
        pass
    else:
        msg = f"Invalid entity type: {entity_type}"
        raise ValueError(msg)
    # Parse binding site specification
    num = len(parsed_chain.residues)

    entry = item[entity_type]
    binding_spec = entry.get("binding_types", None)
    ids = item[entity_type]["id"]
    num_chains = 1 if isinstance(ids, str) else len(ids)
    for _ in range(num_chains):
        if binding_spec is not None:
            if isinstance(binding_spec, str):
                for char in binding_spec:
                    if char.lower() == "u":
                        res_bind_type.append(const.binding_type_ids["UNSPECIFIED"])
                    elif char.lower() == "b":
                        res_bind_type.append(const.binding_type_ids["BINDING"])
                    elif char.lower() == "n":
                        res_bind_type.append(const.binding_type_ids["NOT_BINDING"])
                    else:
                        msg = f"Invalid binding_type '{char}' in: {binding_spec}"
                        raise ValueError(msg)

                # Fill missing specification with unspecified
                if len(binding_spec) < num:
                    num_missing = num - len(binding_spec)
                    res_bind_type.extend(
                        [const.binding_type_ids["UNSPECIFIED"]] * num_missing
                    )
                if len(binding_spec) > num:
                    msg = f"Misspecified bingin_types {binding_spec} which is shorter than the sequence."
                    raise ValueError(msg)
            else:
                types = np.ones(num) * const.binding_type_ids["UNSPECIFIED"]
                if "binding" in binding_spec:
                    indices = parse_range(binding_spec["binding"], 0, num)
                    types[indices] = const.binding_type_ids["BINDING"]

                if "not_binding" in binding_spec:
                    indices = parse_range(binding_spec["not_binding"], 0, num)
                    types[indices] = const.binding_type_ids["NOT_BINDING"]
                res_bind_type.extend(types.tolist())
        else:
            res_bind_type.extend([const.binding_type_ids["UNSPECIFIED"]] * num)

    # Parse ss conditioning specification
    entry = item[entity_type]
    ss_spec = entry.get("secondary_structure", None)
    ids = item[entity_type]["id"]
    num_chains = 1 if isinstance(ids, str) else len(ids)
    for _ in range(num_chains):
        if ss_spec is not None:
            if isinstance(ss_spec, str):
                for char in ss_spec:
                    if char.lower() == "u":
                        ss_type.append(const.ss_type_ids["UNSPECIFIED"])
                    elif char.lower() == "l":
                        ss_type.append(const.ss_type_ids["LOOP"])
                    elif char.lower() == "h":
                        ss_type.append(const.ss_type_ids["HELIX"])
                    elif char.lower() == "s":
                        ss_type.append(const.ss_type_ids["SHEET"])
                    else:
                        msg = f"Invalid secondary_structure '{char}' in: {ss_spec}"
                        raise ValueError(msg)

                # Fill missing specification with unspecified
                if len(ss_spec) < num:
                    num_missing = num - len(ss_spec)
                    ss_type.extend([const.ss_type_ids["UNSPECIFIED"]] * num_missing)
                if len(ss_spec) > num:
                    msg = f"Misspecified secondary_structure {ss_spec} which is shorter than the sequence."
                    raise ValueError(msg)
            else:
                types = np.ones(num) * const.ss_type_ids["UNSPECIFIED"]
                if "loop" in ss_spec:
                    indices = parse_range(ss_spec["loop"], 0, num)
                    types[indices] = const.ss_type_ids["LOOP"]
                if "helix" in ss_spec:
                    indices = parse_range(ss_spec["helix"], 0, num)
                    types[indices] = const.ss_type_ids["HELIX"]
                if "sheet" in ss_spec:
                    indices = parse_range(ss_spec["sheet"], 0, num)
                    types[indices] = const.ss_type_ids["SHEET"]
                ss_type.extend(types.tolist())
        else:
            ss_type.extend([const.ss_type_ids["UNSPECIFIED"]] * num)

    # Add as many parsed_chains as provided ids
    if entity_type in {"protein", "dna", "rna", "ligand"}:
        ids = item[entity_type]["id"]
        if isinstance(ids, str):
            ids = [ids]
        for chain_name in ids:
            parsed_chains[chain_name] = parsed_chain
            chain_to_msa[chain_name] = msa
    fuse = item[entity_type].get("fuse", None)
    fuse_info = {}
    if fuse is not None:
        fuse_info["target_id"] = fuse
        fuse_info["fuse"] = True
    else:
        fuse_info["fuse"] = False
    if is_msa_custom and is_msa_auto:
        msg = "Cannot mix custom and auto-generated MSAs in the same input!"
        raise ValueError(msg)

    return (
        extra_mols,
        parsed_chains,
        res_bind_type,
        ss_type,
        chain_to_msa,
        fuse_info,
        ligand_id,
    )


def parse_redesign_schema(
    schema: dict,
    tokenized: Tokenized,
) -> Target:
    """parse a redesign schema"""
    key = next(iter(schema["restrictions"].keys()))
    if key not in ["not_design", "design"]:
        msg = f"Invalid key: {key}"
        raise ValueError(msg)
    new_design_mask = [False] * len(tokenized.tokens)
    for item in schema["restrictions"][key]:
        # initialize binders to be all designed or num designed
        for token in tokenized.tokens:
            if (
                tokenized.structure.chains[token["asym_id"]]["name"]
                == item["chain"]["binder"]
            ):
                new_design_mask[token["token_idx"]] = key == "not_design"
    for item in schema["restrictions"][key]:
        id = item["chain"]["id"]
        c_start = tokenized.structure.chains[
            np.where(tokenized.structure.chains["name"] == id)
        ][0]["res_idx"].item()
        c_end = (
            c_start
            + tokenized.structure.chains[
                np.where(tokenized.structure.chains["name"] == id)
            ][0]["res_num"].item()
        )
        indicies = parse_range(item["chain"]["res_index"], c_start, c_end)
        token_indices = []
        for idx in range(len(tokenized.tokens)):
            if tokenized.token_to_res[idx] in indicies:
                token_indices.append(idx)
        radius = item["chain"]["within_proximity"]
        undesign_idx = []
        for token in tokenized.tokens:
            for idx in token_indices:
                if (
                    cdist(
                        np.array([token["center_coords"]]),
                        np.array([tokenized.tokens["center_coords"][idx]]),
                    )[0][0]
                    < radius
                    and tokenized.structure.chains[token["asym_id"]]["name"]
                    == item["chain"]["binder"]
                ):
                    undesign_idx.append(token["token_idx"])
                    new_design_mask[token["token_idx"]] = key == "design"
    new_design_mask = np.array(new_design_mask, dtype=bool)
    return new_design_mask


def parse_redesign_yaml(
    path: Path,
    tokenized: Tokenized,
) -> Target:
    """parse a design mask override yaml file"""
    with path.open("r") as file:
        if path.suffix == ".yaml":
            data = yaml.safe_load(file)
        else:
            raise ValueError(f"Unsupported file type: {str(path)}")
    target = parse_redesign_schema(data, tokenized)

    return target


####################################################################################################
# YAML PARSER WRAPPER (with caches)
####################################################################################################


class YamlDesignParser:
    def __init__(
        self,
        mol_dir: Path | str,
    ) -> None:
        self.mol_dir = Path(mol_dir)
        self._struct_cache: dict[tuple[Path, bool], Structure] = {}
        self._once_keys: set[str] = set()

    def parse_yaml(
        self,
        path: Path,
        mols: dict[str, Mol],
        mol_dir: Path,
    ) -> Target:
        """Parse a Boltz input yaml / json."""
        with path.open("r") as file:
            if path.suffix == ".yaml":
                data = yaml.safe_load(file)
            elif path.suffix == ".pdb":
                data = parse_pdb(file)
            else:
                raise ValueError(f"Unsupported file type: {str(path)}")

        name = path.stem
        target = self.parse_boltzgen_schema(
            name, data, mols, mol_dir, base_file_path=path.parent
        )
        return target

    def log_once(self, msg: str):
        """Print *msg* exactly once per *training job* (only on rank-0 if DDP)."""
        try:
            import torch.distributed as _dist

            if _dist.is_available() and _dist.is_initialized():
                is_rank0 = _dist.get_rank() == 0
            else:
                is_rank0 = True
        except Exception:
            is_rank0 = True
        if is_rank0 and msg not in self._once_keys:
            print(msg)
            self._once_keys.add(msg)

    def parse_boltzgen_schema(
        self,
        name: str,
        schema: dict,
        mols: Mapping[str, Mol],
        mol_dir: Optional[Path] = None,
        base_file_path: Optional[Path] = None,
    ) -> Target:
        """Parse a Boltz input yaml / json.
        See examples/design_spec_refactored.yaml for the schema.
        """

        # Check valididty of yaml file
        for name in ["res_idx", "residue_idx", "residue_index"]:
            if name in str(schema):
                raise ValueError(f"Found {name} in yaml. Did you mean 'res_index'?")

        invalid_keys = set()

        def recursive_check(data):
            if isinstance(data, dict):
                for key, value in data.items():
                    if key not in yaml_keys:
                        invalid_keys.add(key)
                    recursive_check(value)
            elif isinstance(data, list):
                for item in data:
                    recursive_check(item)

        recursive_check(schema)
        if len(invalid_keys) > 0:
            msg = f"Found invalid keys in yaml file: {invalid_keys}.\nValid keys are: {yaml_keys}"
            raise ValueError(msg)

        # Disable rdkit warnings
        blocker = rdBase.BlockLogs()  # noqa: F841

        # First group items that have the same type, sequence and modifications
        items_to_group = {}
        file_path_count = {}
        items_list = []
        for item in schema["entities"]:
            # Get entity type
            entity_type = next(iter(item.keys())).lower()
            if entity_type not in {
                "protein",
                "dna",
                "rna",
                "ligand",
                "file",
            }:
                msg = f"Invalid entity type: {entity_type}"
                raise ValueError(msg)

            # Get sequence
            if entity_type in {"protein", "dna", "rna"}:
                seq = str(item[entity_type]["sequence"])
            elif entity_type == "ligand":
                assert "smiles" in item[entity_type] or "ccd" in item[entity_type]
                assert (
                    "smiles" not in item[entity_type] or "ccd" not in item[entity_type]
                )
                if "smiles" in item[entity_type]:
                    seq = str(item[entity_type]["smiles"])
                else:
                    seq = str(item[entity_type]["ccd"])
            elif entity_type == "file":
                identifier = str(item["file"]["path"])
                file_path_count[identifier] = file_path_count.get(identifier, 0) + 1
                seq = identifier + str(file_path_count[identifier])
            items_list.append(item)
            items_to_group.setdefault((entity_type, seq), []).append(item)

            # Create tables

        protein_chains = set()

        covalents = []
        constraints = schema.get("constraints", [[]])
        if "total_len" in constraints[0]:
            total_len = constraints[0]["total_len"]
            if "min" in total_len:
                min_len = total_len["min"]
            if "max" in total_len:
                max_len = total_len["max"]

        # Convert parsed chains to tables

        while True:
            data = Structure.empty_protein(0)

            chain_to_idx = {}

            # Keep a mapping of (chain_name, residue_idx, atom_name) to atom_idx
            atom_idx_map = {}
            local_atom_idx_map = {}
            total_renaming = {}
            extra_mols = {}
            res_bind_type = []
            ss_type = []
            chain_to_msa = {}

            is_msa_custom = False
            is_msa_auto = False
            all_parsed_chains: dict[str, ParsedChain] = {}
            ligand_id = 1
            structure_groups = np.array([], dtype=np.int32)
            res_design_mask = np.array([], dtype=bool)
            res_bind_type = np.array([], dtype=np.int32)
            ss_type = np.array([], dtype=np.int32)
            chain_to_msa = {}

            global_asym_id = 0
            for item in items_list:
                sym_id = 0
                entity_type = next(iter(item.keys())).lower()
                if entity_type != "file":
                    atom_idx = 0
                    res_idx = 0
                    asym_id = 0
                    atom_data = []
                    bond_data = []
                    res_data = []
                    chain_data = []
                    new_res_design_mask = []
                    (
                        new_extra_mols,
                        parsed_chains,
                        new_res_bind_type,
                        new_ss_type,
                        entity_chain_to_msa,
                        fuse_info,
                        ligand_id,
                    ) = parse_entity(
                        item, mols, mol_dir, ligand_id, is_msa_custom, is_msa_auto
                    )
                    all_parsed_chains.update(parsed_chains)

                    extra_mols.update(new_extra_mols)
                    res_bind_type = np.concatenate([res_bind_type, new_res_bind_type])
                    ss_type = np.concatenate([ss_type, new_ss_type])
                    for asym_id, (chain_name, chain) in enumerate(
                        parsed_chains.items()
                    ):
                        # Compute number of atoms and residues
                        res_num = len(chain.residues)
                        atom_num = sum(len(res.atoms) for res in chain.residues)

                        # Extend res_design_mask
                        new_res_design_mask.extend(chain.res_design_mask)

                        # Save protein chains for later
                        if chain.type == const.chain_type_ids["PROTEIN"]:
                            protein_chains.add(chain_name)

                        # Find all copies of this chain in the assembly
                        chain_data.append(
                            (
                                chain_name,
                                chain.type,
                                0,
                                sym_id,
                                asym_id,
                                atom_idx,
                                atom_num,
                                res_idx,
                                res_num,
                                chain.cyclic_period,
                            )
                        )
                        chain_to_idx[chain_name] = asym_id
                        sym_id += 1

                        # Add residue, atom, bond, data
                        for res in chain.residues:
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

                            for bond in res.bonds:
                                atom_1 = atom_idx + bond.atom_1
                                atom_2 = atom_idx + bond.atom_2
                                bond_data.append(
                                    (
                                        asym_id,
                                        asym_id,
                                        res_idx,
                                        res_idx,
                                        atom_1,
                                        atom_2,
                                        bond.type,
                                    )
                                )

                            for atom in res.atoms:
                                # Add atom to map
                                atom_idx_map[(chain_name, res.idx, atom.name)] = (
                                    global_asym_id,
                                    data.residues.shape[0]
                                    + asym_id * res_num
                                    + res_idx,
                                    data.atoms.shape[0] + asym_id * atom_num + atom_idx,
                                )
                                local_atom_idx_map[(chain_name, res.idx, atom.name)] = (
                                    asym_id,
                                    asym_id * res_num + res_idx,
                                    asym_id * atom_num + atom_idx,
                                )

                                # Add atom to data
                                atom_data.append(
                                    (
                                        atom.name,
                                        atom.element,
                                        atom.charge,
                                        atom.coords,
                                        atom.conformer,
                                        atom.is_present,
                                        atom.chirality,
                                    )
                                )
                                atom_idx += 1
                            res_idx += 1
                        if chain.cyclic_period > 0:
                            bond_data.append(
                                (
                                    asym_id,
                                    asym_id,
                                    0,
                                    chain.cyclic_period - 1,
                                    local_atom_idx_map[(chain_name, 0, "N")][2],
                                    local_atom_idx_map[
                                        (chain_name, chain.cyclic_period - 1, "C")
                                    ][2],
                                    const.bond_type_ids["COVALENT"],
                                )
                            )

                    new_res_design_mask = np.array(new_res_design_mask)
                    residues = np.array(res_data, dtype=Residue)
                    chains = np.array(chain_data, dtype=Chain)
                    interfaces = np.array([], dtype=Interface)
                    mask = np.ones(len(chain_data), dtype=bool)
                    atom_data = [(a[0], a[3], a[5], 0.0, 1.0) for a in atom_data]
                    atoms = np.array(atom_data, dtype=Atom)
                    bonds = np.array(bond_data, dtype=Bond)
                    coords = [(x,) for x in atoms["coords"]]
                    coords = np.array(coords, Coords)
                    ensemble = np.array([(0, len(coords))], dtype=Ensemble)
                    new_data = Structure(
                        atoms=atoms,
                        bonds=bonds,
                        residues=residues,
                        chains=chains,
                        interfaces=interfaces,
                        mask=mask,
                        coords=coords,
                        ensemble=ensemble,
                    )
                    new_structure_groups = np.zeros(
                        len(new_data.residues), dtype=np.int32
                    )
                    structure_groups = np.concatenate(
                        [structure_groups, new_structure_groups]
                    )
                    res_design_mask = np.concatenate(
                        [res_design_mask, new_res_design_mask]
                    )
                    if fuse_info["fuse"]:
                        data = Structure.fuse(
                            data, new_data, fuse_info["target_id"], res_reindex=True
                        )
                        msg = f"fused chain{fuse_info['target_id']} with chain{new_data.chains[0]['name']}"
                        self.log_once(msg)
                    else:
                        data, renaming = Structure.concatenate(
                            data, new_data, return_renaming=True
                        )
                        total_renaming.update(renaming)
                        if len(renaming) > 0:
                            msg = f"\nChain ids in non-file sequence conflict with existing chain ids. Renaming them {renaming}."
                            self.log_once(msg)
                        global_asym_id += asym_id + 1
                    new_chain_to_msa = {}
                    for chain_id, msa in entity_chain_to_msa.items():
                        renamed_id = renaming.get(chain_id, chain_id)
                        if renamed_id in chain_to_msa:
                            raise KeyError(
                                f"Key '{renamed_id}' already exists in chain_to_msa."
                            )
                        new_chain_to_msa[renamed_id] = msa
                    chain_to_msa.update(new_chain_to_msa)
                else:
                    path = item["file"]["path"]
                    (
                        new_data,
                        new_groups,
                        new_design_mask,
                        fbind_types,
                        fss_type,
                        file_chain_to_msa,
                        fuse_info,
                        new_extra_mols,
                        file_msa_flag,
                        ligand_id,
                    ) = self.parse_file(item, mols, mol_dir, ligand_id, base_file_path)
                    if fuse_info["fuse"]:
                        if fuse_info["target_id"] in total_renaming.keys():
                            fuse_info["target_id"] = total_renaming[
                                fuse_info["target_id"]
                            ]
                        data = Structure.fuse(
                            data, new_data, fuse_info["target_id"], res_reindex=True
                        )
                        msg = f"\nFused chain{fuse_info['target_id']} with chain{new_data.chains[0]['name']}."
                        self.log_once(msg)
                    else:
                        data, renaming = Structure.concatenate(
                            data, new_data, return_renaming=True
                        )
                        total_renaming.update(renaming)
                        global_asym_id += max(new_data.chains["asym_id"]) + 1
                    structure_groups = np.concatenate([structure_groups, new_groups])
                    res_design_mask = np.concatenate([res_design_mask, new_design_mask])
                    res_bind_type = np.concatenate([res_bind_type, fbind_types])
                    ss_type = np.concatenate([ss_type, fss_type])
                    extra_mols.update(new_extra_mols)
                    if len(renaming) > 0:
                        msg = f"\nChain ids conflict with existing chain ids. Renaming with {renaming}. This is for the structure from '{path}'."
                        self.log_once(msg)
                    new_chain_to_msa = {}
                    for chain_id, msa in file_chain_to_msa.items():
                        renamed_id = renaming.get(chain_id, chain_id)
                        if renamed_id in chain_to_msa:
                            raise KeyError(
                                f"Key '{renamed_id}' already exists in chain_to_msa."
                            )
                        new_chain_to_msa[renamed_id] = msa
                    chain_to_msa.update(new_chain_to_msa)
                    # Update chain_to_msa dictionary. Set defaults given by file_msa_flag for proteins. Insert -1 (no msa) for {dna, rna, ligand}.
                    for chain in data.chains:
                        chain_id = chain["name"].item()
                        if chain_id not in chain_to_msa:
                            if chain["mol_type"] == const.chain_type_ids["PROTEIN"]:
                                chain_to_msa[chain_id] = file_msa_flag
                            else:
                                chain_to_msa[chain_id] = -1
            if "total_len" in constraints[0]:
                if len(res_bind_type) >= min_len and len(res_bind_type) <= max_len:
                    break
            if "total_len" not in constraints[0]:
                break

        # Parse constraints
        for constraint in constraints:
            if "bond" in constraint:
                if (
                    "atom1" not in constraint["bond"]
                    or "atom2" not in constraint["bond"]
                ):
                    msg = f"Bond constraint was not properly specified"
                    raise ValueError(msg)

                c1, r1, a1 = tuple(constraint["bond"]["atom1"])
                c2, r2, a2 = tuple(constraint["bond"]["atom2"])
                r1 = r1 - 1  # 1-indexed
                r2 = r2 - 1  # 1-indexed
                if c1 in total_renaming.keys():
                    c1 = total_renaming[c1]
                if c2 in total_renaming.keys():
                    c2 = total_renaming[c2]

                if c1 not in all_parsed_chains.keys():
                    msg = f"Chain {c1} in the specified connection does not exist: {constraint}"
                    ValueError(msg)
                if c2 not in all_parsed_chains.keys():
                    msg = f"Chain {c2} in the specified connection does not exist: {constraint}"
                    ValueError(msg)

                # Map index
                if (
                    c1 in all_parsed_chains.keys()
                    and all_parsed_chains[c1].sampleidx_to_specidx is not None
                ):
                    r1 = np.where(all_parsed_chains[c1].sampleidx_to_specidx == r1)[0][
                        0
                    ].item()
                    c1, r1, a1 = atom_idx_map[(c1, r1, a1)]
                else:
                    # we have a chain coming from a file where we just use the residue index
                    chain = data.chains[data.chains["name"] == c1]
                    c1 = chain["asym_id"].item()

                    res_start = chain["res_idx"].item()
                    res_end = chain["res_idx"].item() + chain["res_num"].item()
                    residues = data.residues[res_start:res_end]
                    residue = residues[residues["res_idx"] == r1]
                    r1 = res_start + residue["res_idx"].item()

                    atom_start = residue["atom_idx"].item()
                    atom_end = residue["atom_idx"].item() + residue["atom_num"].item()
                    atoms = data.atoms[atom_start:atom_end]
                    assert a1 in atoms["name"], (
                        f"Atom {a1} not found in residue {r1} of chain {c1}"
                    )
                    a1 = np.where(atoms["name"] == a1)[0].item()
                    a1 = (
                        residue["atom_idx"].item() + a1
                    )  # THIS STILL NEEDS TO BE CORRECTED

                if (
                    c2 in all_parsed_chains.keys()
                    and all_parsed_chains[c2].sampleidx_to_specidx is not None
                ):
                    r2 = np.where(all_parsed_chains[c2].sampleidx_to_specidx == r2)[0][
                        0
                    ].item()
                    c2, r2, a2 = atom_idx_map[(c2, r2, a2)]
                else:
                    # we have a chain coming from a file where we just use the residue index
                    chain = data.chains[data.chains["name"] == c2]
                    c2 = chain["asym_id"].item()

                    res_start = chain["res_idx"].item()
                    res_end = chain["res_idx"].item() + chain["res_num"].item()
                    residues = data.residues[res_start:res_end]
                    residue = residues[residues["res_idx"] == r2]
                    r2 = res_start + residue["res_idx"].item()

                    atom_start = residue["atom_idx"].item()
                    atom_end = residue["atom_idx"].item() + residue["atom_num"].item()
                    atoms = data.atoms[atom_start:atom_end]
                    assert a2 in atoms["name"], (
                        f"Atom {a2} not found in residue {r2} of chain {c2}"
                    )
                    a2 = np.where(atoms["name"] == a2)[0].item()
                    a2 = (
                        residue["atom_idx"].item() + a2
                    )  # THIS STILL NEEDS TO BE CORRECTED
                covalents.append((c1, c2, r1, r2, a1, a2))
            elif "total_len" in constraints:
                continue

        covalents = [(*c, const.bond_type_ids["COVALENT"]) for c in covalents]
        covalents = np.array(covalents, dtype=Bond)
        data = replace(data, bonds=np.concatenate([data.bonds, covalents]))

        # Parse leaving atoms
        leaving_atoms = schema.get("leaving_atoms", [])
        for leaving_atom in leaving_atoms:
            cidx, ridx, aidx = tuple(leaving_atom["atom"])
            ridx = ridx - 1
            if all_parsed_chains[cidx].sampleidx_to_specidx is not None:
                ridx = np.where(all_parsed_chains[cidx].sampleidx_to_specidx == ridx)[
                    0
                ][0].item()
            if cidx in total_renaming.keys():
                cidx = total_renaming[cidx]
            chain = data.chains[np.where(data.chains["name"] == cidx)[0].item()]
            residues = data.residues[
                chain["res_idx"] : chain["res_idx"] + chain["res_num"]
            ]
            res = residues[np.where(residues["res_idx"] == ridx)[0].item()]
            atoms = data.atoms[res["atom_idx"] : res["atom_idx"] + res["atom_num"]]
            atom_idx = res["atom_idx"] + np.where(atoms["name"] == aidx)[0].item()
            data.atoms["is_present"][atom_idx] = False

        # Create metadata
        struct_info = StructureInfo(num_chains=len(data.chains))
        chain_infos = []
        for chain in data.chains:
            chain_info = ChainInfo(
                chain_id=int(chain["asym_id"]),
                chain_name=chain["name"],
                mol_type=int(chain["mol_type"]),
                cluster_id=-1,
                msa_id=chain_to_msa[chain["name"]],
                num_residues=int(chain["res_num"]),
                valid=True,
                entity_id=int(chain["entity_id"]),
            )
            chain_infos.append(chain_info)

        record = Record(
            id=name,
            structure=struct_info,
            chains=chain_infos,
            interfaces=[],
        )

        design_info = DesignInfo(
            res_design_mask=res_design_mask,
            res_structure_groups=structure_groups,
            res_binding_type=res_bind_type,
            res_ss_types=ss_type,
        )
        DesignInfo.is_valid(design_info)

        return Target(
            record=record,
            structure=data,
            design_info=design_info,
            extra_mols=extra_mols,
        )

    def parse_file(self, item, mols, mol_dir, ligand_id, base_file_path=Path(".")):
        extra_mols: dict[str, Mol] = {}

        file = item["file"]

        # Check if file points to another yaml file. If so, then use the contents of that other yaml file
        path = file["path"]
        if isinstance(path, list) or Path(path).suffix == ".yaml":
            if isinstance(path, list):
                path = random.choice(path)

            resolved_path = (base_file_path / path).resolve()
            with resolved_path.open("r") as f:
                file = yaml.safe_load(f)
                base_file_path = resolved_path.parent

        # Extract values of file
        path = (base_file_path / Path(file["path"])).resolve()
        use_assembly = file.get("use_assembly", False)  # dont use assembly by default
        include = file.get("include", "all")  # include all by default
        include_proximity = file.get("include_proximity", None)
        exclude = file.get("exclude", None)
        structure_spec = file.get("structure_groups", None)
        design = file.get("design", None)
        add_cyclization = file.get("add_cyclization", None)
        reset_res_index = file.get("reset_res_index", None)
        not_design = file.get("not_design", None)
        file_msa_flag = file.get("msa", 0)  # default to automatic MSA generation
        if (file_msa_flag is None) or (file_msa_flag == ""):
            file_msa_flag = 0
        design_insertions = file.get("design_insertions", None)
        fuse = file.get("fuse", None)
        binding_types = file.get("binding_types", None)
        secondary_structure = file.get("secondary_structure", None)

        if isinstance(include, list):
            for list_element in include:
                chain = list_element["chain"]
                if "id" not in chain:
                    msg = f"Misspecified chain in include with missing 'id' for file with path {path}."
                    raise ValueError(msg)
                chain_id = chain["id"]
                if "smiles" in chain:
                    mol = AllChem.MolFromSmiles(chain["smiles"])
                    mol = AllChem.AddHs(mol)
                    element_counts = defaultdict(int)
                    for i, atom in enumerate(mol.GetAtoms()):
                        symbol = atom.GetSymbol()
                        element_counts[symbol] += 1
                        atom_name = f"{symbol.upper()}{element_counts[symbol]}"
                        atom.SetProp("name", atom_name)
                    mols[f"LIG{ligand_id}"] = mol

                    success = compute_3d_conformer(mol)

                    if not success:
                        msg = f"Failed to compute 3D conformer for given smiles string"
                        raise ValueError(msg)
                    extra_mols[f"LIG{ligand_id}"] = mol
                    ligand_id += 1

        # Get structure
        cache_key = (path.resolve(), use_assembly)
        cached = self._struct_cache.get(cache_key)

        if cached is not None:
            parsed = deepcopy(cached)
        else:
            if path.suffix == ".pdb":
                parsed = parse_pdb(
                    path,
                    mols=mols,
                    moldir=mol_dir,
                    use_assembly=use_assembly,
                )
            else:
                parsed = parse_mmcif(
                    path,
                    mols=mols,
                    moldir=mol_dir,
                    use_assembly=use_assembly,
                )
            self._struct_cache[cache_key] = deepcopy(parsed)

        structure = parsed.data
        num_res = len(structure.residues)

        # Construct include mask from include entries
        file_chain_to_msa = {}
        if isinstance(include, str):
            if include == "all":
                include_mask = np.ones(num_res)
            else:
                msg = f"Include has to be a list or 'all' to include everything in the file."
                raise ValueError(msg)
        elif isinstance(include, list):
            include_mask = np.zeros(num_res)
            for list_element in include:
                chain = list_element["chain"]
                if "id" not in chain:
                    msg = f"Misspecified chain in include with missing 'id' for file with path {path}."
                    raise ValueError(msg)
                chain_id = chain["id"]
                if chain_id not in structure.chains["name"]:
                    msg = f"Specified chain id {chain_id} not in file {path}."
                    raise ValueError(msg)

                if "msa" in chain:
                    file_chain_to_msa[chain_id] = chain["msa"]
                data_chain = structure.chains[structure.chains["name"] == chain_id]

                c_start = data_chain["res_idx"].item()
                c_end = c_start + data_chain["res_num"].item()

                # Set include_mask values to 1
                if "res_index" not in chain:
                    include_mask[c_start:c_end] = 1
                else:
                    indices = parse_range(chain["res_index"], c_start, c_end)
                    include_mask[indices] = 1
        else:
            msg = "Include entry has to be a list of chains or 'all'."
            raise ValueError(msg)

        proximity_mask = np.ones(num_res)
        if include_proximity is not None:
            proximity_mask = np.zeros(num_res)
            coords = np.array(
                [
                    structure.atoms[r["atom_center"]]["coords"]
                    for r in structure.residues
                ]
            )

            for list_element in include_proximity:
                chain = list_element["chain"]
                if "id" not in chain:
                    msg = f"Misspecified chain in include_proximity with missing 'id' for file with path {path}."
                    raise ValueError(msg)
                if "radius" not in chain:
                    msg = f"Misspecified chain in include_proximity with missing 'radius' for file with path {path}."
                    raise ValueError(msg)
                chain_id = chain["id"]
                radius = chain["radius"]
                if chain_id not in structure.chains["name"]:
                    msg = f"Specified chain id {chain_id} not in file {path}."
                    raise ValueError(msg)

                data_chain = structure.chains[structure.chains["name"] == chain_id]
                c_start = data_chain["res_idx"].item()
                c_end = c_start + data_chain["res_num"].item()

                proximity_spec_mask = np.zeros(num_res)
                if "res_index" not in chain:
                    proximity_spec_mask[c_start:c_end] = 1
                else:
                    indices = parse_range(chain["res_index"], c_start, c_end)
                    proximity_spec_mask[indices] = 1

                queries = coords[proximity_spec_mask.astype(bool)]

                distances = cdist(coords, queries)
                dist_mask = distances < radius
                dist_mask = dist_mask.sum(-1) > 0
                proximity_mask += dist_mask
        include_mask *= proximity_mask

        # Build exclude mask
        exclude_mask = np.ones(num_res)
        if exclude is not None:
            for list_element in exclude:
                chain = list_element["chain"]
                if "id" not in chain:
                    msg = f"Misspecified chain in exclude with missing 'id' for file with path {path}."
                    raise ValueError(msg)
                chain_id = chain["id"]
                if chain_id not in structure.chains["name"]:
                    msg = f"Specified chain id {chain_id} not in file {path}."
                    raise ValueError(msg)

                data_chain = structure.chains[structure.chains["name"] == chain_id]
                c_start = data_chain["res_idx"].item()
                c_end = c_start + data_chain["res_num"].item()

                # Set exclude_mask values to 0
                if "res_index" not in chain:
                    include_mask[c_start:c_end] = 0
                else:
                    indices = parse_range(chain["res_index"], c_start, c_end)
                    exclude_mask[indices] = 0
        include_mask = (include_mask * exclude_mask).astype(bool)

        # Construct missing_mask to remove leading and trailing unresolved residues for every chain.
        missing_mask = []
        for chain in structure.chains:
            chain_start = chain["res_idx"]
            chain_end = chain["res_idx"] + chain["res_num"]
            chain_include_mask = include_mask[chain_start:chain_end]
            if chain_include_mask.sum() == 0:
                # Just append ones for the whole chain if the chain is not even included. This will leave the include_mask unaffected.
                missing_mask.append(np.ones(chain["res_num"], dtype=bool))
            else:
                # Make missing_mask of trailing and leading residues in the included part of the chain
                chain_res = structure.residues[chain_start:chain_end]
                included_res = chain_res[chain_include_mask]
                is_present = included_res["is_present"]
                first_true = np.argmax(is_present)
                last_true = len(is_present) - 1 - np.argmax(is_present[::-1])
                included_missing_mask = np.ones_like(is_present, dtype=bool)
                included_missing_mask[:first_true] = False
                included_missing_mask[last_true + 1 :] = False

                # Print a message if there are any trailing or leading missing residues.
                if (~included_missing_mask).sum() > 0:
                    if included_missing_mask.sum() == 0:
                        msg = f"\nThere are no resolved residues for chain {chain['name']} in {str(path)}. We are removing the chain."
                    else:
                        leading = ",".join(map(str, included_res[:first_true]["name"]))
                        trailing = ",".join(
                            map(str, included_res[last_true + 1 :]["name"])
                        )
                        msg = (
                            f"\nRemoving leading and/or trailing unresolved residues from included part of chain {chain['name']} in {path}.\n"
                            f"  Leading unresolved:  {leading}\n"
                            f"  Trailing unresolved: {trailing}"
                        )
                    self.log_once(msg)

                # insert the missing mask of the included part of the chain into the missing mask of the whole chain
                chain_missing_mask = np.ones(chain["res_num"], dtype=bool)
                chain_missing_mask[chain_include_mask] = included_missing_mask
                missing_mask.append(chain_missing_mask)
        missing_mask = np.concatenate(missing_mask)
        include_mask *= missing_mask

        # Get structure groups
        new_groups = np.zeros(num_res)
        if structure_spec is None or structure_spec == "all" or structure_spec == 1:
            new_groups = np.ones(num_res)
        else:
            for list_element in structure_spec:
                group = list_element["group"]
                if "id" not in group:
                    msg = f"Misspecified group in structure_groups with missing 'id' for file with path {path}."
                    raise ValueError(msg)
                if "visibility" not in group:
                    msg = f"Misspecified group in structure_groups with missing 'visibility' for file with path {path}."
                    raise ValueError(msg)
                chain_id = group["id"]

                # Handle the "all" case where all chains are set to be specified
                if chain_id == "all":
                    new_groups = np.ones(num_res)
                    continue

                if chain_id not in structure.chains["name"]:
                    msg = f"Specified chain id {chain_id} not in file {path}."
                    raise ValueError(msg)

                data_chain = structure.chains[structure.chains["name"] == chain_id]
                c_start = data_chain["res_idx"].item()
                c_end = c_start + data_chain["res_num"].item()
                visibility = group["visibility"]

                # Set structure group values to the correct visibility
                if "res_index" not in group:
                    new_groups[c_start:c_end] = visibility
                else:
                    indices = parse_range(group["res_index"], c_start, c_end)
                    new_groups[indices] = visibility

        # Get design mask for file
        new_design_mask = np.zeros(num_res)
        if design is not None:
            for list_element in design:
                chain = list_element["chain"]
                if "id" not in chain:
                    msg = f"Misspecified chain in design with missing 'id' for file with path {path}."
                    raise ValueError(msg)
                chain_id = chain["id"]

                # Handle the "all" case where all chains are set to be designed
                if chain_id == "all":
                    # TODO: handle case where users specify non-protein residues to be designed.
                    new_design_mask = np.ones(num_res)
                    continue

                if chain_id not in structure.chains["name"]:
                    msg = f"Specified chain id {chain_id} not in file {path}."
                    raise ValueError(msg)

                data_chain = structure.chains[structure.chains["name"] == chain_id]
                c_start = data_chain["res_idx"].item()
                c_end = c_start + data_chain["res_num"].item()

                # Set values
                if "res_index" not in chain:
                    new_design_mask[c_start:c_end] = 1
                else:
                    indices = parse_range(chain["res_index"], c_start, c_end)
                    new_design_mask[indices] = 1

        # Get modification mask to turn previous design regions into non-design regions
        new_design_mask_mod = np.ones(num_res)
        if not_design is not None:
            for list_element in not_design:
                chain = list_element["chain"]
                if "id" not in chain:
                    msg = f"Misspecified chain in not_design with missing 'id' for file with path {path}."
                    raise ValueError(msg)
                chain_id = chain["id"]

                if chain_id not in structure.chains["name"]:
                    msg = f"Specified chain id {chain_id} not in file {path}."
                    raise ValueError(msg)

                data_chain = structure.chains[structure.chains["name"] == chain_id]
                c_start = data_chain["res_idx"].item()
                c_end = c_start + data_chain["res_num"].item()

                # Set values
                if "res_index" not in chain:
                    new_design_mask_mod[c_start:c_end] = 0
                else:
                    indices = parse_range(chain["res_index"], c_start, c_end)
                    new_design_mask_mod[indices] = 0
        new_design_mask = (new_design_mask * new_design_mask_mod).astype(bool)

        # Get file's binding types called fbind_types
        fbind_types = np.ones(num_res) * const.binding_type_ids["UNSPECIFIED"]
        fbind_types = fbind_types.astype(np.int32)
        if binding_types is not None:
            for list_element in binding_types:
                chain = list_element["chain"]
                if "id" not in chain:
                    msg = f"Misspecified chain in binding_types with missing 'id' for file with path {path}."
                    raise ValueError(msg)
                chain_id = chain["id"]

                if chain_id not in structure.chains["name"]:
                    msg = f"Specified chain id {chain_id} not in file {path}."
                    raise ValueError(msg)

                data_chain = structure.chains[structure.chains["name"] == chain_id]
                c_start = data_chain["res_idx"].item()
                c_end = c_start + data_chain["res_num"].item()

                # Set values
                if "not_binding" in chain:
                    not_binding = chain["not_binding"]
                    if not_binding == "all":
                        fbind_types[c_start:c_end] = const.binding_type_ids[
                            "NOT_BINDING"
                        ]
                    else:
                        indices = parse_range(not_binding, c_start, c_end)
                        fbind_types[indices] = const.binding_type_ids["NOT_BINDING"]
                elif "binding" in chain:
                    binding = chain["binding"]
                    if binding == "all":
                        fbind_types[c_start:c_end] = const.binding_type_ids["BINDING"]
                    else:
                        indices = parse_range(binding, c_start, c_end)
                        fbind_types[indices] = const.binding_type_ids["BINDING"]

        # Get file's secondary structure types called fss_types
        fss_type = np.ones(num_res) * const.ss_type_ids["UNSPECIFIED"]
        fss_type = fss_type.astype(np.int32)
        if secondary_structure is not None:
            for list_element in secondary_structure:
                chain = list_element["chain"]
                if "id" not in chain:
                    msg = f"Misspecified chain in secondary_structure with missing 'id' for file with path {path}."
                    raise ValueError(msg)
                chain_id = chain["id"]

                if chain_id not in structure.chains["name"]:
                    msg = f"Specified chain id {chain_id} not in file {path}."
                    raise ValueError(msg)

                data_chain = structure.chains[structure.chains["name"] == chain_id]
                c_start = data_chain["res_idx"].item()
                c_end = c_start + data_chain["res_num"].item()

                # Set values
                if "loop" in chain:
                    loop = chain["loop"]
                    if loop == "all":
                        fss_type[c_start:c_end] = const.ss_type_ids["LOOP"]
                    else:
                        indices = parse_range(loop, c_start, c_end)
                        fss_type[indices] = const.ss_type_ids["LOOP"]
                if "helix" in chain:
                    helix = chain["helix"]
                    if helix == "all":
                        fss_type[c_start:c_end] = const.ss_type_ids["HELIX"]
                    else:
                        indices = parse_range(helix, c_start, c_end)
                        fss_type[indices] = const.ss_type_ids["HELIX"]
                if "sheet" in chain:
                    sheet = chain["sheet"]
                    if sheet == "all":
                        fss_type[c_start:c_end] = const.ss_type_ids["SHEET"]
                    else:
                        indices = parse_range(sheet, c_start, c_end)
                        fss_type[indices] = const.ss_type_ids["SHEET"]

        # Parse and apply design insertions
        if design_insertions is not None:
            num_inserted = defaultdict(int)
            for list_element in design_insertions:
                insertion = list_element["insertion"]
                if "id" not in insertion:
                    msg = f"Misspecified insertion in design_insertions with missing 'id' for file with path {path}."
                    raise ValueError(msg)
                if "res_index" not in insertion:
                    msg = f"Misspecified insertion in design_insertions with missing 'res_index' for file with path {path}."
                    raise ValueError(msg)
                chain_id = insertion["id"]
                res_index = insertion["res_index"] - 1  # 1 index input to 0 indexed
                res_index += num_inserted[chain_id]
                ss_insert_type = insertion.get("secondary_structure", "UNSPECIFIED")

                # We add +1 because the parse_range function is usually used for indexing where we then convert the 1 based inputs to 0 indexing
                num_residues = insertion["num_residues"]
                num_residues = parse_range(num_residues)
                num_residues = np.random.choice(num_residues).item()
                num_residues += 1
                num_inserted[chain_id] += num_residues

                if chain_id not in structure.chains["name"]:
                    msg = f"Specified chain id {chain_id} not in file {path}."
                    raise ValueError(msg)

                target_chain = structure.chains[structure.chains["name"] == chain_id]
                res_insert_idx = target_chain["res_idx"] + res_index

                # Insert into structure
                structure = Structure.insert(
                    structure, chain_id, res_idx=res_index, num_residues=num_residues
                )

                # Insert into design specifications
                include_mask = np.insert(
                    include_mask, res_insert_idx, np.ones(num_residues)
                )
                new_groups = np.insert(
                    new_groups, res_insert_idx, np.zeros(num_residues)
                )
                new_design_mask = np.insert(
                    new_design_mask, res_insert_idx, np.ones(num_residues)
                )
                fbind_types = np.insert(
                    fbind_types,
                    res_insert_idx,
                    np.ones(num_residues) * const.binding_type_ids["UNSPECIFIED"],
                )
                fss_type = np.insert(
                    fss_type,
                    res_insert_idx,
                    np.ones(num_residues) * const.ss_type_ids[ss_insert_type],
                )

        # Apply mask to new structure groups. Update structure_groups by concatenating existing and new one
        new_groups = new_groups[include_mask].astype(np.int32)

        # Apply mask to new design_mask. Update design by concatenating existing and new one.
        new_design_mask = new_design_mask[include_mask]

        # Apply mask to new binding_types. Update binding_types by concatenating existing and new one.
        fbind_types = fbind_types[include_mask].astype(np.int32)

        # Apply mask to new ss_type. Update ss_type by concatenating existing and new one.
        fss_type = fss_type[include_mask].astype(np.int32)

        # Apply mask to structrue
        if not all(include_mask):
            new_structure = Structure.extract_residues(
                structure, include_mask.astype(bool), res_reindex=False
            )
        else:
            new_structure = structure

        # Handle cyclizations
        if add_cyclization is not None:
            additional_bonds = []
            for list_element in add_cyclization:
                chain = list_element["chain"]
                if "id" not in chain:
                    msg = f"Misspecified chain in add_cyclization with missing 'id' for file with path {path}."
                    raise ValueError(msg)
                chain_id = chain["id"]
                chain_idx = np.where(chain_id == new_structure.chains["name"])[0].item()
                struct_chain = new_structure.chains[chain_idx]
                num_res = struct_chain["res_num"].item()
                new_structure.chains[chain_idx]["cyclic_period"] = num_res
                chain_res_idx = struct_chain["res_idx"].item()

                # Get atom indices
                res1 = new_structure.residues[chain_res_idx]
                res2 = new_structure.residues[chain_res_idx + num_res - 1]
                atoms1 = new_structure.atoms[
                    res1["atom_idx"] : res1["atom_idx"] + res1["atom_num"]
                ]
                atoms2 = new_structure.atoms[
                    res2["atom_idx"] : res2["atom_idx"] + res2["atom_num"]
                ]
                assert "N" in atoms1["name"]
                assert "C" in atoms2["name"]
                idx_in_res1 = np.where(atoms1["name"] == "N")[0].item()
                idx_in_res2 = np.where(atoms2["name"] == "C")[0].item()
                atom_idx1 = res1["atom_idx"] + idx_in_res1
                atom_idx2 = res2["atom_idx"] + idx_in_res2

                # Make new bond
                additional_bonds.append(
                    (
                        struct_chain["asym_id"].item(),
                        struct_chain["asym_id"].item(),
                        chain_res_idx,
                        chain_res_idx + num_res - 1,
                        atom_idx1,
                        atom_idx2,
                        const.bond_type_ids["COVALENT"],
                    )
                )
            additional_bonds = np.array(additional_bonds, dtype=Bond)
            new_bonds = np.concatenate([new_structure.bonds, additional_bonds])
            new_structure = replace(new_structure, bonds=new_bonds)

        # Reset residue indices of chains where it is desired
        if reset_res_index is not None:
            for list_element in reset_res_index:
                chain = list_element["chain"]
                if "id" not in chain:
                    msg = f"Misspecified chain in reset_res_index with missing 'id' for file with path {path}."
                    raise ValueError(msg)
                chain_id = chain["id"]
                chain_idx = np.where(chain_id == new_structure.chains["name"])[0].item()
                struct_chain = new_structure.chains[chain_idx]
                new_structure.residues[
                    struct_chain["res_idx"] : struct_chain["res_idx"]
                    + struct_chain["res_num"]
                ]["res_idx"] = np.arange(struct_chain["res_num"])

        # perform fusion or concatenation
        fuse_info = {}
        if fuse is not None:
            fuse_info["target_id"] = file["fuse"]
            fuse_info["fuse"] = True
        else:
            fuse_info["fuse"] = False

        return (
            new_structure,
            new_groups,
            new_design_mask,
            fbind_types,
            fss_type,
            file_chain_to_msa,
            fuse_info,
            extra_mols,
            file_msa_flag,
            ligand_id,
        )
