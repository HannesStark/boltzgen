"""Residue-normalized ipSAE, with the conservative minimum of both directions.

The directional d0res calculation follows DunbrackLab/IPSAE at 6174cf9e.
Ligand atoms are excluded. Modified polymers contribute one CA/C1' token per
residue, just like canonical polymers. No coordinate-distance cutoff is used.
"""

import numpy as np


def directed_ipsae(pae: np.ndarray, cutoff: float = 10.0, d0_min: float = 1.0) -> float:
    """Maximum row score using the number of low-PAE partners in each row."""
    if pae.size == 0:
        return 0.0
    valid = pae < cutoff
    count = valid.sum(axis=1)
    d0 = np.maximum(d0_min, 1.24 * np.cbrt(np.maximum(count, 19) - 15) - 1.8)
    terms = 1 / (1 + (pae / d0[:, None]) ** 2)
    scores = np.divide(
        np.where(valid, terms, 0).sum(axis=1),
        count,
        out=np.zeros(len(count)),
        where=count > 0,
    )
    return float(scores.max(initial=0))


def score_interface(
    pae: np.ndarray, design: list[int], target: list[int], *, nucleic_acid: bool = False
) -> dict[str, float]:
    """Score two disjoint sets of polymer residue representatives."""
    pae = np.asarray(pae, dtype=np.float64)
    if (
        pae.ndim != 2
        or pae.shape[0] != pae.shape[1]
        or not np.isfinite(pae).all()
        or (pae < 0).any()
    ):
        raise ValueError("ipSAE requires a square, finite, nonnegative PAE matrix")
    for indices in (design, target):
        if len(indices) != len(set(indices)) or any(
            i < 0 or i >= len(pae) for i in indices
        ):
            raise ValueError(
                "ipSAE residue indices must be unique and within the PAE matrix"
            )
    if set(design) & set(target):
        raise ValueError("Design and target must be disjoint partners")
    floor = 2.0 if nucleic_acid else 1.0
    forward = directed_ipsae(pae[np.ix_(design, target)], d0_min=floor)
    reverse = directed_ipsae(pae[np.ix_(target, design)], d0_min=floor)
    return {
        "esmfold2_ipsae_min": min(forward, reverse),
        "esmfold2_design_to_target_ipsae": forward,
        "esmfold2_target_to_design_ipsae": reverse,
    }


def score_chain_vs_rest(
    pae: np.ndarray, representatives: dict[str, list[int]], kinds: dict[str, int]
) -> dict[str, dict[str, float]]:
    """Score each polymer chain against all other polymers in a redesign."""
    if len(representatives) < 2:
        raise ValueError("Chain-versus-rest ipSAE requires at least two polymer chains")
    return {
        chain: score_interface(
            pae,
            indices,
            [
                i
                for other, positions in representatives.items()
                if other != chain
                for i in positions
            ],
            nucleic_acid=any(kinds[other] in (1, 2) for other in representatives),
        )
        for chain, indices in representatives.items()
    }
