import warnings
import string
from typing import Mapping, Optional

import gemmi
import numpy as np
from rdkit.Chem.rdchem import Mol

from boltzgen.data.parse.mmcif import (
    mmcif_from_block,
    ParsedStructure,
)
from boltzgen.data.data import (
    Chain,
    Structure,
)


def parse_pdb(  # noqa: C901, PLR0915, PLR0912
    path: str,
    mols: Mapping[str, Mol] = {},
    moldir: Optional[str] = None,
    use_assembly: bool = True,
    use_original_res_idx: bool = False,
) -> ParsedStructure:
    """
    WARNING: this function does not parse ligands if they are seperated with TER records.
    """

    if not str(path).endswith(".pdb"):
        warnings.warn(f"Input file {path} does not have .pdb extension")

    st = gemmi.read_structure(str(path))
    st.setup_entities()

    for entity in st.entities:
        if "Polymer" != entity.entity_type.name:
            continue
        sc = entity.subchains[0]
        sc = st[0].get_subchain(sc)

        # If full_sequence does not exist, we create it starting from 1 in the subchain.
        # This is needed for some .pdb files that are generated from other sources (e.g. solubleMPNN).
        if len(entity.full_sequence) < 1:
            full_sequence = [res.name for res in sc]
            entity.full_sequence = full_sequence

        # Manual label_seq matching with alignment.
        align_result = gemmi.align_sequence_to_polymer(
            entity.full_sequence,
            sc,
            entity.polymer_type,
            gemmi.AlignmentScoring(),
        ).match_string

        i = 0
        for j, align in enumerate(align_result):
            if align == "|":
                sc[i].label_seq = j + 1
                i += 1

        # Fix: Set label_seq for all subchains in the same entity, not just the first one
        # This is important when multiple chains (e.g., A, B, C) belong to the same entity
        for subchain_id in entity.subchains[1:]:  # Skip the first one, already processed
            subchain = st[0].get_subchain(subchain_id)
            if subchain is not None:
                # Use the same alignment result for all subchains in the same entity
                i = 0
                for j, align in enumerate(align_result):
                    if align == "|" and i < len(subchain):
                        subchain[i].label_seq = j + 1
                        i += 1

    block = st.make_mmcif_block()

    structure = mmcif_from_block(
        block, mols, moldir, use_assembly, use_original_res_idx=use_original_res_idx
    )

    # block = st.make_mmcif_block() changes chain names from A to Axp etc because MMCIF allows for
    # 3-letter chain names and PDB only allows for 1-letter chain names.
    # This call patches the chain names back to the original 1-letter names
    # NOTE A better solution could be to change the chain names in the block directly, before
    # calling mmcif_from_block. The authentic names should be also stored there somewhere but they
    # are not used.
    chain_names = [chain.name for chain in st[0]]
    if any(len(chain) > 1 for chain in chain_names):
        raise ValueError(
            f"Chain names in the PDB file are not 1-letter: {chain_names}. This should never happen"
            "according to PDB spec. But it is also possible to extend patch_chain_names with an"
            "additional argument to pass the original chain names to distinguish between chains"
            "with the same first letter, and patch chain names unambiguously."
        )
    structure = patch_chain_names(structure)

    return structure


def patch_chain_names(structure: ParsedStructure) -> ParsedStructure:
    """Patch chain names to first character according to PDB spec. Ligands within same chain id
    get renamed to avoid ambiguity.
    """
    # Patch chain names
    chains_new = []
    chain_mapping = {}
    chain_names = [chain["name"][0] for chain in structure.data.chains]
    chain_id_pool = list(reversed(string.ascii_uppercase)) + list(
        reversed(string.digits)
    )
    used_names = []
    for chain in structure.data.chains:
        chain_name = chain["name"][0]
        if chain_name in used_names:
            for candidate in chain_id_pool:
                if candidate not in chain_names and candidate not in used_names:
                    chain_name = candidate
                    break
        used_names.append(chain_name)
        chain_mapping[chain["name"]] = chain_name
        chains_new.append(
            (
                chain_name,
                chain["mol_type"],
                chain["entity_id"],
                chain["sym_id"],
                chain["asym_id"],
                chain["atom_idx"],
                chain["atom_num"],
                chain["res_idx"],
                chain["res_num"],
                chain["cyclic_period"],
                chain["symmetric_group"],
            )
        )
    chains_new = np.array(chains_new, dtype=Chain)

    # Patch sequences
    sequences_new = {
        chain_mapping[name]: seq for name, seq in structure.sequences.items()
    }

    data = Structure(
        atoms=structure.data.atoms,
        bonds=structure.data.bonds,
        residues=structure.data.residues,
        chains=chains_new,
        interfaces=structure.data.interfaces,
        mask=structure.data.mask,
        ensemble=structure.data.ensemble,
        coords=structure.data.coords,
    )
    return ParsedStructure(
        data=data,
        info=structure.info,
        sequences=sequences_new,
    )
