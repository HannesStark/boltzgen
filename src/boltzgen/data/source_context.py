"""Full polymer context that survives design, spatial cropping, and inverse folding."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import gemmi
import numpy as np

from boltzgen.data.data import Structure


def from_structure(structure: Structure, source: str = "inline") -> dict:
    """Capture residue identities before any spatial selection or renumbering."""
    chains = []
    for chain in structure.chains:
        start, count = int(chain["res_idx"]), int(chain["res_num"])
        names = structure.residues["name"][start : start + count].tolist()
        chains.append(
            {
                "source_chain": str(chain["name"]),
                "source": source,
                "mol_type": int(chain["mol_type"]),
                "residue_names": names,
                "indices": list(range(count)),
                "complete": True,
            }
        )
    return {"version": 1, "chains": chains}


def from_file(structure: Structure, path: Path, overrides: list | None) -> dict:
    """Require declared polymer sequences, or an explicit full-sequence mapping."""
    context = from_structure(structure, str(path))
    source = gemmi.read_structure(str(path))
    # Do not synthesize SEQRES here: the ordinary Boltz parser does, and doing so
    # would make an already cropped PDB look like a complete source chain.
    if path.suffix == ".pdb":
        # This matches parse_pdb's entity ordering but does not synthesize any
        # sequence. Assembly copies and renamed chains retain this entity ID.
        source.setup_entities()
    for chain, parsed_chain in zip(context["chains"], structure.chains, strict=True):
        if chain["mol_type"] != 3:
            entity = source.entities[int(parsed_chain["entity_id"])]
            # Match the existing structure parser's MSE -> MET normalization.
            declared = tuple(
                "MET" if name == "MSE" else name for name in entity.full_sequence
            )
            chain["complete"] = (
                bool(declared) and tuple(chain["residue_names"]) == declared
            )
    for entry in overrides or []:
        spec = entry["chain"]
        matches = [c for c in context["chains"] if c["source_chain"] == spec["id"]]
        if len(matches) != 1:
            raise ValueError(f"Full-sequence chain {spec['id']} is absent or ambiguous")
        chain = matches[0]
        sequence = spec["sequence"]
        letters = {
            0: dict(
                zip(
                    "ARNDCQEGHILKMFPSTWYV",
                    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
                )
            ),
            1: {x: "D" + x for x in "ACGT"},
            2: {x: x for x in "ACGU"},
        }
        try:
            full_names = [letters[chain["mol_type"]][aa] for aa in sequence]
        except KeyError as exc:
            raise ValueError(
                "full_sequences requires a canonical polymer sequence"
            ) from exc
        positions = spec.get("source_res_indices")
        if positions is None:
            raise ValueError(
                "full_sequences requires one source_res_indices entry per input residue"
            )
        if any(type(i) is not int for i in positions):
            raise ValueError("source_res_indices must contain integer positions")
        indices = [i - 1 for i in positions]
        if (
            len(indices) != len(chain["indices"])
            or indices != sorted(set(indices))
            or not indices
            or indices[0] < 0
            or indices[-1] >= len(full_names)
        ):
            raise ValueError(
                "source_res_indices must be unique, increasing, in-range 1-based positions"
            )
        for name, index in zip(chain["residue_names"], indices):
            if name != full_names[index]:
                raise ValueError(
                    f"Full sequence disagrees with {spec['id']} at position {index + 1}"
                )
        chain.update(residue_names=full_names, indices=indices, complete=True)
    return context


def replacement_labels(
    structure: Structure, excluded: np.ndarray, designed: np.ndarray
) -> np.ndarray:
    """Label fully designable, contiguous exclusion intervals within each chain."""
    labels = np.full(len(structure.residues), -1, dtype=np.int64)
    for chain in structure.chains:
        start, count = int(chain["res_idx"]), int(chain["res_num"])
        positions = np.flatnonzero(excluded[start : start + count]) + start
        intervals = np.split(positions, np.flatnonzero(np.diff(positions) != 1) + 1)
        for interval in intervals:
            if interval.size and designed[interval].all():
                labels[interval] = interval[0]
    return labels


def select(
    context: dict,
    structure: Structure,
    mask: np.ndarray,
    replaced: np.ndarray | None = None,
) -> dict:
    """Keep full crop context, removing only excluded residues being replaced."""
    result = deepcopy(context)
    chains = []
    for entry, chain in zip(result["chains"], structure.chains, strict=True):
        start, count = int(chain["res_idx"]), int(chain["res_num"])
        selected = [
            i
            for i, keep in zip(
                entry["indices"], mask[start : start + count], strict=True
            )
            if keep
        ]
        if replaced is not None:
            removed = {
                i for i, remove in zip(
                    entry["indices"], replaced[start : start + count], strict=True
                ) if remove
            }
            if removed:
                assert removed.isdisjoint(selected)
                # Keep the sequence and map before removal as provenance. The
                # generated sequence later updates only the edited full context.
                entry.setdefault("replacement_sources", []).append({
                    "source": entry["source"],
                    "source_chain": entry["source_chain"],
                    "residue_names": entry["residue_names"],
                    "indices": selected,
                })
                kept = [i for i in range(len(entry["residue_names"])) if i not in removed]
                new_index = {old: new for new, old in enumerate(kept)}
                entry["residue_names"] = [entry["residue_names"][i] for i in kept]
                selected = [new_index[i] for i in selected]
        entry["indices"] = selected
        if entry["indices"]:
            chains.append(entry)
    result["chains"] = chains
    return result


def insert(context: dict, chain_id: str, position: int, count: int) -> None:
    """Insert sampled design residues into the full construct and its residue map."""
    entry = next(c for c in context["chains"] if c["source_chain"] == chain_id)
    old = entry["indices"]
    full_position = old[position] if position < len(old) else old[-1] + 1
    entry["residue_names"][full_position:full_position] = ["GLY"] * count
    shifted = [i + count if i >= full_position else i for i in old]
    shifted[position:position] = range(full_position, full_position + count)
    entry["indices"] = shifted


def merge(context: dict, incoming: dict, before: Structure, fusion: str | None) -> dict:
    """Preserve source context, or assemble the construct explicitly declared by fuse."""
    result = deepcopy(context)
    if fusion is None:
        result["chains"].extend(deepcopy(incoming["chains"]))
        return result
    index = before.chains["name"].tolist().index(fusion)
    left = result["chains"][index]
    (right,) = incoming["chains"]
    # Unlike spatial selection alone, fuse explicitly joins the selected
    # segments into one construct. Its ESMC sequence follows that assembly.
    left["residue_names"] = [left["residue_names"][i] for i in left["indices"]] + [
        right["residue_names"][i] for i in right["indices"]
    ]
    left["indices"] = list(range(len(left["residue_names"])))
    left["complete"] = left["complete"] and right["complete"]
    left["context_mode"] = "fused_construct"
    left["source"] += "+" + right["source"]
    if right.get("replacement_sources"):
        left.setdefault("replacement_sources", []).extend(right["replacement_sources"])
    return result


def update_designed(context: dict, structure: Structure) -> dict:
    """Insert the generated sequence into the preserved full-source sequence."""
    result = deepcopy(context)
    for entry, chain in zip(result["chains"], structure.chains, strict=True):
        start, count = int(chain["res_idx"]), int(chain["res_num"])
        names = structure.residues["name"][start : start + count].tolist()
        for index, name in zip(entry["indices"], names, strict=True):
            entry["residue_names"][index] = name
        entry["output_chain"] = str(chain["name"])
    return result
