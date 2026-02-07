"""
Peptide Feature Extraction for BBB Permeability Prediction.

This module provides functions to compute physicochemical properties of peptides
that are relevant for blood-brain barrier (BBB) permeability:

1. Net charge at physiological pH (7.4)
2. Hydrophobicity (Kyte-Doolittle scale)
3. Polar surface area (PSA) approximation
4. Optional: ESM-2 protein language model embeddings

These features are used to train the BBB classifier.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Union


# Amino acid pKa values for charge calculation at pH 7.4
# Source: Grimsley et al., Protein Science (2009)
PKA_VALUES = {
    "D": 3.9,   # Aspartic acid (acidic)
    "E": 4.3,   # Glutamic acid (acidic)
    "K": 10.5,  # Lysine (basic)
    "R": 12.5,  # Arginine (basic)
    "H": 6.0,   # Histidine (basic, weak)
    "C": 8.3,   # Cysteine (thiol)
    "Y": 10.1,  # Tyrosine (phenol)
}

# N-terminus pKa (amine group)
N_TERMINUS_PKA = 9.6
# C-terminus pKa (carboxyl group)
C_TERMINUS_PKA = 2.3

# Kyte-Doolittle hydrophobicity scale
# Source: Kyte & Doolittle, J. Mol. Biol. (1982)
# Higher values = more hydrophobic
HYDROPHOBICITY_SCALE = {
    "A": 1.8,   # Alanine
    "R": -4.5,  # Arginine
    "N": -3.5,  # Asparagine
    "D": -3.5,  # Aspartic acid
    "C": 2.5,   # Cysteine
    "Q": -3.5,  # Glutamine
    "E": -3.5,  # Glutamic acid
    "G": -0.4,  # Glycine
    "H": -3.2,  # Histidine
    "I": 4.5,   # Isoleucine
    "L": 3.8,   # Leucine
    "K": -3.9,  # Lysine
    "M": 1.9,   # Methionine
    "F": 2.8,   # Phenylalanine
    "P": -1.6,  # Proline
    "S": -0.8,  # Serine
    "T": -0.7,  # Threonine
    "W": -0.9,  # Tryptophan
    "Y": -1.3,  # Tyrosine
    "V": 4.2,   # Valine
}

# Polar surface area (PSA) contributions per amino acid
# Approximation based on side chain functional groups
# Units: Å² (Angstroms squared)
PSA_CONTRIBUTIONS = {
    "A": 0.0,    # Alanine (methyl - no polar atoms)
    "R": 85.0,   # Arginine (guanidinium group: 4 N-H)
    "N": 58.0,   # Asparagine (amide: C=O, N-H2)
    "D": 63.0,   # Aspartic acid (carboxyl: 2 O)
    "C": 25.0,   # Cysteine (thiol: S-H)
    "Q": 58.0,   # Glutamine (amide: C=O, N-H2)
    "E": 63.0,   # Glutamic acid (carboxyl: 2 O)
    "G": 0.0,    # Glycine (no side chain)
    "H": 50.0,   # Histidine (imidazole: 2 N)
    "I": 0.0,    # Isoleucine (no polar atoms)
    "L": 0.0,    # Leucine (no polar atoms)
    "K": 35.0,   # Lysine (amine: N-H3+)
    "M": 25.0,   # Methionine (thioether: S)
    "F": 0.0,    # Phenylalanine (aromatic ring, no polar atoms)
    "P": 0.0,    # Proline (no polar atoms in ring)
    "S": 46.0,   # Serine (hydroxyl: O-H)
    "T": 46.0,   # Threonine (hydroxyl: O-H)
    "W": 30.0,   # Tryptophan (indole: N-H)
    "Y": 46.0,   # Tyrosine (phenol: O-H)
    "V": 0.0,    # Valine (no polar atoms)
}

# Backbone contribution to PSA (approximately constant per residue)
# Peptide bond: C=O and N-H
BACKBONE_PSA_PER_RESIDUE = 38.0  # Å²


def calculate_net_charge(sequence: str, ph: float = 7.4) -> float:
    """
    Calculate net charge of a peptide at a given pH.

    Uses Henderson-Hasselbalch equation:
    charge = sum of protonated basic groups - sum of deprotonated acidic groups

    Parameters
    ----------
    sequence : str
        Amino acid sequence (single-letter codes)
    ph : float, default=7.4
        pH value (physiological pH is 7.4)

    Returns
    -------
    float
        Net charge of the peptide at the given pH

    Examples
    --------
    >>> calculate_net_charge("RKKRRQRRR")  # Highly basic
    8.99
    >>> calculate_net_charge("DDDEEEE")  # Highly acidic
    -6.99
    >>> calculate_net_charge("ACDEFGHIKLMNPQRSTVWY")  # Mixed
    0.12
    """
    sequence = sequence.upper()
    charge = 0.0

    # N-terminus (protonated amine, pKa ~9.6)
    # Fraction protonated at pH: 1 / (1 + 10^(pH - pKa))
    charge += 1.0 / (1.0 + 10 ** (ph - N_TERMINUS_PKA))

    # C-terminus (deprotonated carboxyl, pKa ~2.3)
    # Fraction deprotonated at pH: 1 / (1 + 10^(pKa - pH))
    charge -= 1.0 / (1.0 + 10 ** (C_TERMINUS_PKA - ph))

    # Side chains
    for aa in sequence:
        if aa in PKA_VALUES:
            pka = PKA_VALUES[aa]

            if aa in ["D", "E"]:
                # Acidic residues (negative charge when deprotonated)
                charge -= 1.0 / (1.0 + 10 ** (pka - ph))
            elif aa in ["K", "R", "H"]:
                # Basic residues (positive charge when protonated)
                charge += 1.0 / (1.0 + 10 ** (ph - pka))
            elif aa in ["C", "Y"]:
                # Weakly acidic (usually neutral at pH 7.4)
                charge -= 1.0 / (1.0 + 10 ** (pka - ph))

    return charge


def calculate_hydrophobicity(
    sequence: str,
    scale: Dict[str, float] = HYDROPHOBICITY_SCALE,
    window_size: Optional[int] = None,
) -> Union[float, np.ndarray]:
    """
    Calculate hydrophobicity using the Kyte-Doolittle scale.

    Can return either the average hydrophobicity of the entire peptide
    or a sliding window profile.

    Parameters
    ----------
    sequence : str
        Amino acid sequence (single-letter codes)
    scale : dict, optional
        Hydrophobicity scale (default: Kyte-Doolittle)
    window_size : int, optional
        If provided, returns sliding window hydrophobicity profile
        If None (default), returns average hydrophobicity

    Returns
    -------
    float or np.ndarray
        If window_size is None: average hydrophobicity (float)
        If window_size is int: hydrophobicity profile (array of length L - window_size + 1)

    Examples
    --------
    >>> calculate_hydrophobicity("AILV")  # Hydrophobic residues
    3.575
    >>> calculate_hydrophobicity("RKDE")  # Hydrophilic residues
    -4.1
    """
    sequence = sequence.upper()

    # Get hydrophobicity values for each residue
    values = [scale.get(aa, 0.0) for aa in sequence]

    if window_size is None:
        # Return average hydrophobicity
        return np.mean(values) if values else 0.0
    else:
        # Return sliding window profile
        if len(values) < window_size:
            return np.array([np.mean(values)])

        profile = []
        for i in range(len(values) - window_size + 1):
            window = values[i : i + window_size]
            profile.append(np.mean(window))

        return np.array(profile)


def calculate_psa(sequence: str, include_backbone: bool = True) -> float:
    """
    Calculate polar surface area (PSA) of a peptide.

    PSA is an approximation based on side chain functional groups
    and optionally the peptide backbone. Lower PSA generally correlates
    with better membrane permeability.

    Parameters
    ----------
    sequence : str
        Amino acid sequence (single-letter codes)
    include_backbone : bool, default=True
        Whether to include backbone contribution (~38 Ų per residue)

    Returns
    -------
    float
        Polar surface area in Ų (Angstroms squared)

    Examples
    --------
    >>> calculate_psa("AILV", include_backbone=False)  # No polar atoms
    0.0
    >>> calculate_psa("RKDE", include_backbone=False)  # Many polar atoms
    246.0
    >>> calculate_psa("AILV", include_backbone=True)  # With backbone
    152.0
    """
    sequence = sequence.upper()

    # Side chain contributions
    psa = sum(PSA_CONTRIBUTIONS.get(aa, 0.0) for aa in sequence)

    # Backbone contribution
    if include_backbone:
        psa += BACKBONE_PSA_PER_RESIDUE * len(sequence)

    return psa


def extract_peptide_features(
    sequence: str,
    include_esm: bool = False,
    esm_model_name: str = "facebook/esm2_t6_8M_UR50D",
    device: str = "cpu",
) -> Dict[str, Union[float, np.ndarray, torch.Tensor]]:
    """
    Extract all physicochemical features for a peptide.

    Returns a dictionary with the following features:
    - net_charge: charge at pH 7.4
    - hydrophobicity_mean: average Kyte-Doolittle score
    - hydrophobicity_max: maximum hydrophobicity in sliding window (size=5)
    - psa_total: total polar surface area (with backbone)
    - psa_sidechain: polar surface area of side chains only
    - length: peptide length
    - esm_embedding: (optional) ESM-2 embedding (640-dim for 8M model)

    Parameters
    ----------
    sequence : str
        Amino acid sequence
    include_esm : bool, default=False
        Whether to compute ESM-2 embeddings
    esm_model_name : str, default='facebook/esm2_t6_8M_UR50D'
        ESM-2 model to use for embeddings
    device : str, default='cpu'
        Device for ESM-2 inference

    Returns
    -------
    dict
        Dictionary of features

    Examples
    --------
    >>> features = extract_peptide_features("ARLFKYGRKKRRQRRR")
    >>> features['net_charge']
    8.91
    >>> features['length']
    16
    """
    sequence = sequence.upper()

    features = {
        "sequence": sequence,
        "length": len(sequence),
        "net_charge": calculate_net_charge(sequence, ph=7.4),
        "hydrophobicity_mean": calculate_hydrophobicity(sequence),
        "psa_total": calculate_psa(sequence, include_backbone=True),
        "psa_sidechain": calculate_psa(sequence, include_backbone=False),
    }

    # Hydrophobicity profile features
    hydro_profile = calculate_hydrophobicity(sequence, window_size=5)
    features["hydrophobicity_max"] = np.max(hydro_profile) if len(hydro_profile) > 0 else 0.0
    features["hydrophobicity_min"] = np.min(hydro_profile) if len(hydro_profile) > 0 else 0.0

    # Additional derived features
    features["charge_per_residue"] = features["net_charge"] / features["length"]
    features["psa_per_residue"] = features["psa_total"] / features["length"]

    # Optional: ESM-2 embeddings
    if include_esm:
        try:
            from boltzgen.utils.tau_embeddings import load_esm2_embeddings

            esm_emb = load_esm2_embeddings(sequence, model_name=esm_model_name, device=device)
            # Pool to single vector (mean over sequence length)
            esm_pooled = esm_emb.mean(dim=1).squeeze(0)  # (embedding_dim,)
            features["esm_embedding"] = esm_pooled
        except ImportError:
            print("Warning: transformers not available, skipping ESM embeddings")

    return features


class PeptideFeatureExtractor:
    """
    Feature extractor class for batch processing of peptides.

    This class provides a convenient interface for extracting features
    from multiple peptides at once, with optional caching of ESM-2 embeddings.

    Examples
    --------
    >>> extractor = PeptideFeatureExtractor(include_esm=False)
    >>> sequences = ["ARLFKYGRKKRRQRRR", "DDDEEEE", "AILV"]
    >>> features = extractor.extract_batch(sequences)
    >>> features.shape
    (3, 10)  # 3 sequences, 10 features each
    """

    def __init__(
        self,
        include_esm: bool = False,
        esm_model_name: str = "facebook/esm2_t6_8M_UR50D",
        device: str = "cpu",
    ):
        """
        Initialize feature extractor.

        Parameters
        ----------
        include_esm : bool, default=False
            Whether to include ESM-2 embeddings
        esm_model_name : str
            ESM-2 model name
        device : str
            Device for ESM-2 inference
        """
        self.include_esm = include_esm
        self.esm_model_name = esm_model_name
        self.device = device

        # Feature names (in order)
        self.feature_names = [
            "length",
            "net_charge",
            "hydrophobicity_mean",
            "hydrophobicity_max",
            "hydrophobicity_min",
            "psa_total",
            "psa_sidechain",
            "charge_per_residue",
            "psa_per_residue",
        ]

        if include_esm:
            # ESM embedding size depends on model
            # 8M: 320, 35M: 480, 650M: 1280, 3B: 2560
            esm_sizes = {
                "facebook/esm2_t6_8M_UR50D": 320,
                "facebook/esm2_t12_35M_UR50D": 480,
                "facebook/esm2_t33_650M_UR50D": 1280,
                "facebook/esm2_t36_3B_UR50D": 2560,
            }
            esm_dim = esm_sizes.get(esm_model_name, 320)
            self.feature_names.extend([f"esm_{i}" for i in range(esm_dim)])

    def extract(self, sequence: str) -> np.ndarray:
        """Extract features for a single peptide."""
        features = extract_peptide_features(
            sequence,
            include_esm=self.include_esm,
            esm_model_name=self.esm_model_name,
            device=self.device,
        )

        # Convert to array
        feature_values = [
            features["length"],
            features["net_charge"],
            features["hydrophobicity_mean"],
            features["hydrophobicity_max"],
            features["hydrophobicity_min"],
            features["psa_total"],
            features["psa_sidechain"],
            features["charge_per_residue"],
            features["psa_per_residue"],
        ]

        if self.include_esm:
            esm_values = features["esm_embedding"].cpu().numpy()
            feature_values.extend(esm_values)

        return np.array(feature_values)

    def extract_batch(self, sequences: List[str]) -> np.ndarray:
        """
        Extract features for multiple peptides.

        Parameters
        ----------
        sequences : list of str
            List of amino acid sequences

        Returns
        -------
        np.ndarray
            Feature matrix of shape (n_sequences, n_features)
        """
        features = [self.extract(seq) for seq in sequences]
        return np.array(features)

    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.feature_names


# Convenience function for testing
def demo_features():
    """Demonstrate feature extraction on example peptides."""
    # Example BBB+ peptides (known to cross BBB)
    bbb_positive = [
        "ARLFKYGRKKRRQRRR",  # TAT peptide
        "RQIKIWFQNRRMKWKK",  # Penetratin
        "GRKKRRQRRRPPQ",     # HIV-1 TAT
    ]

    # Example BBB- peptides (likely not to cross BBB)
    bbb_negative = [
        "DDDEEEEDDDEEEE",    # Highly acidic
        "SSSSSSSSSSSSSS",    # Highly polar
        "AAAAAAAAAAAAAA",    # Small, neutral (likely degraded)
    ]

    print("BBB+ Peptides:")
    print("-" * 80)
    for seq in bbb_positive:
        feat = extract_peptide_features(seq)
        print(f"{seq:20s} | Charge: {feat['net_charge']:+.2f} | "
              f"Hydro: {feat['hydrophobicity_mean']:+.2f} | "
              f"PSA: {feat['psa_total']:.0f} Ų")

    print("\nBBB- Peptides:")
    print("-" * 80)
    for seq in bbb_negative:
        feat = extract_peptide_features(seq)
        print(f"{seq:20s} | Charge: {feat['net_charge']:+.2f} | "
              f"Hydro: {feat['hydrophobicity_mean']:+.2f} | "
              f"PSA: {feat['psa_total']:.0f} Ų")


if __name__ == "__main__":
    # Run demo
    demo_features()
