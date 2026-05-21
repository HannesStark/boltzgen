from __future__ import annotations

from Bio.PDB import MMCIFParser, PDBParser


def _load_structure(path: str):
    if path.lower().endswith(".cif"):
        return MMCIFParser(QUIET=True).get_structure("x", path)
    return PDBParser(QUIET=True).get_structure("x", path)


def score_structure(path: str, binder_chain: str = "B", target_chain: str = "A", cutoff: float = 5.0) -> dict:
    structure = _load_structure(path)
    model = next(structure.get_models())
    binder = model[binder_chain]
    target = model[target_chain]

    contacts = set()
    for b_res in binder:
        for t_res in target:
            for ba in b_res.get_atoms():
                if any((ba - ta) <= cutoff for ta in t_res.get_atoms()):
                    contacts.add((b_res.id[1], t_res.id[1]))
                    break

    bset = {c[0] for c in contacts}
    tset = {c[1] for c in contacts}
    return {
        "num_contacts": len(contacts),
        "binder_contact_residues": len(bset),
        "target_contact_residues": len(tset),
        "contact_pairs": sorted(contacts),
    }
