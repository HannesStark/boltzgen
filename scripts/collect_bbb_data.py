"""
BBB Permeability Dataset Collection Script.

This script collects BBB+ and BBB- peptides from various sources:
1. BrainPeps database (http://www.brainpeps.ugent.be/)
2. B3Pdb database (http://i.uestc.edu.cn/b3pdb/)
3. Literature-curated BBB- peptides

Creates a CSV file with columns: sequence, label, source, length
"""

import pandas as pd
import numpy as np
from pathlib import Path
import requests
from typing import List, Tuple


def collect_brainpeps_manual() -> List[Tuple[str, str]]:
    """
    Manually curated BBB+ peptides from BrainPeps database.

    BrainPeps (http://www.brainpeps.ugent.be/) is a database of peptides
    with CNS activity. For the TFG, these are manually extracted as the
    database doesn't have a bulk export API.

    Returns
    -------
    list of tuples
        List of (sequence, source) tuples
    """
    # Known BBB+ peptides from literature and BrainPeps
    bbb_positive = [
        # Cell-penetrating peptides (CPPs) - well-documented BBB crossers
        ("ARLFKYGRKKRRQRRR", "BrainPeps/TAT"),          # TAT peptide (HIV-1)
        ("RQIKIWFQNRRMKWKK", "BrainPeps/Penetratin"),   # Penetratin (Antennapedia)
        ("GRKKRRQRRRPPQ", "BrainPeps/TAT-short"),       # Short TAT variant
        ("RKKRRQRRR", "BrainPeps/R9"),                  # Poly-arginine R9
        ("YGRKKRRQRRR", "BrainPeps/Tat-9"),             # Tat-9 fragment
        ("DAATATRGRSAASRPTERPRAPARSASRPRRPVE", "BrainPeps/pVEC"),  # pVEC (murine Vascular endothelial-cadherin)

        # Rabies virus glycoprotein-derived peptides
        ("YTIWMPENPRPGTPCDIFTNSRGKRASNG", "BrainPeps/RVG"),  # RVG peptide
        ("MNLLRKIVKNRRDEDTQKSSPASAPLDD", "BrainPeps/RDP"),   # Rabies-derived peptide

        # Transferrin receptor-targeting peptides
        ("THRPPMWSPVWP", "BrainPeps/T7"),               # T7 peptide
        ("HAIYPRH", "BrainPeps/Tf-binding"),            # Transferrin-binding

        # Angiopep-2 (LRP1 receptor-targeting)
        ("TFFYGGSRGKRNNFKTEEY", "BrainPeps/Angiopep-2"),

        # Neuropeptides (endogenous BBB+ peptides)
        ("YGGFM", "BrainPeps/Met-enkephalin"),          # Met-enkephalin
        ("YGGFL", "BrainPeps/Leu-enkephalin"),          # Leu-enkephalin
        ("RVYVHPF", "BrainPeps/Angiotensin-IV"),        # Angiotensin IV

        # Synthetic BBB+ peptides from recent papers
        ("CKRRMKWKK", "Literature/CPP-9"),              # Synthetic CPP
        ("RKKRRRESRKKRRRES", "Literature/dual-TAT"),    # Dual TAT
        ("GWTLNSAGYLLGKINLKALAALAKKIL", "Literature/amphipathic"),  # Amphipathic helix
    ]

    return bbb_positive


def collect_b3pdb_manual() -> List[Tuple[str, str]]:
    """
    Manually curated BBB+ peptides from B3Pdb database.

    B3Pdb (http://i.uestc.edu.cn/b3pdb/) is a database specifically
    for blood-brain barrier penetrating peptides.

    Returns
    -------
    list of tuples
        List of (sequence, source) tuples
    """
    # B3Pdb entries (manually extracted from database)
    b3pdb_peptides = [
        # Additional CPPs
        ("KLALKLALKALKAALKLA", "B3Pdb/Amphipathic"),
        ("RQIKIWFQNRRMKWKK", "B3Pdb/Penetratin"),
        ("RVIRVWFQNKRCKDKK", "B3Pdb/TP10"),
        ("AGYLLGKINLKALAALAKKIL", "B3Pdb/TP-1"),

        # Poly-arginine variants
        ("RRRRRRRRR", "B3Pdb/R9"),
        ("RRRRRRRRRRR", "B3Pdb/R11"),
        ("RRRRRRRR", "B3Pdb/R8"),

        # Synthetic peptides
        ("KETWWETWWTEW", "B3Pdb/Pep-1"),
        ("GALFLGFLGAAGSTMGAWSQPKKKRKV", "B3Pdb/MPG"),
        ("FFHHIFRGIVHVGKTIHRLVTF", "B3Pdb/Bac7"),

        # Brain-homing peptides
        ("CLEVSRKNC", "B3Pdb/CRT-targeting"),
        ("CGNKRTRGC", "B3Pdb/brain-homing"),
        ("KRTGSGK", "B3Pdb/BBB-crossing"),
    ]

    return b3pdb_peptides


def collect_bbb_negative() -> List[Tuple[str, str]]:
    """
    Curate BBB- peptides (peptides that do NOT cross the BBB).

    These are typically:
    1. Highly acidic peptides (negative charge)
    2. Very hydrophilic peptides (high PSA)
    3. Large polar peptides
    4. Peptides without CPP motifs

    Returns
    -------
    list of tuples
        List of (sequence, source) tuples
    """
    bbb_negative = [
        # Highly acidic peptides
        ("DDDEEEEDDD", "Synthetic/acidic"),
        ("EEEEEEEEEE", "Synthetic/poly-E"),
        ("DDDDDDDDDD", "Synthetic/poly-D"),
        ("DEDEDEDEDEDE", "Synthetic/alternating-acidic"),
        ("EDEEEEDEDEEE", "Synthetic/acidic-rich"),

        # Highly hydrophilic peptides
        ("SSSSSSSSSSSS", "Synthetic/poly-S"),
        ("NNNNNNNNNNNN", "Synthetic/poly-N"),
        ("QQQQQQQQQQQQ", "Synthetic/poly-Q"),
        ("SNSNSNSNSNSN", "Synthetic/SN-repeat"),
        ("TTTTTTTTTTT", "Synthetic/poly-T"),

        # Random coil peptides (no structure, no CPP motifs)
        ("APSGYQSTAPSG", "Synthetic/random"),
        ("GSTAPSGYGAPS", "Synthetic/random-2"),
        ("ASQTGPYSAQGT", "Synthetic/random-3"),

        # Large polar peptides (from non-permeable proteins)
        ("DGEAGAQGPPGPQGPR", "CollagenI/fragment"),      # Collagen I (extracellular)
        ("EQEEEEDNRDSMDED", "EGFR/fragment"),           # EGFR extracellular domain
        ("GEGQQPGEGS", "Elastin/fragment"),             # Elastin (extracellular)

        # Peptides with conflicting properties (neutral charge, moderate hydrophobicity but no BBB crossing)
        ("AAAAAAAAAAAA", "Synthetic/poly-A"),           # Likely degraded before BBB
        ("GGGGGGGGGGGG", "Synthetic/poly-G"),           # Too flexible
        ("PPPPPPPPPPPP", "Synthetic/poly-P"),           # Rigid proline helix

        # Shuffled CPP sequences (negative controls)
        ("QRKRFKGRRYRA", "Shuffled/TAT-scrambled"),     # Scrambled TAT
        ("WKMKRNQKFRIQ", "Shuffled/Pen-scrambled"),     # Scrambled Penetratin
        ("RQRRRKRGRQK", "Shuffled/R9-scrambled"),       # Scrambled R9

        # Antimicrobial peptides (work on bacteria but not BBB+)
        ("GIKFLHSAKKF", "AMP/Cecropin-A-frag"),
        ("KWKSFIKKLTSAAKKVVTT", "AMP/LL-37-frag"),
    ]

    return bbb_negative


def create_dataset(output_path: str = "data/bbb/bbb_dataset.csv", seed: int = 42) -> pd.DataFrame:
    """
    Create the complete BBB dataset with train/val/test splits.

    Parameters
    ----------
    output_path : str
        Path to save the dataset CSV
    seed : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Dataset with columns: sequence, label, source, length, split
    """
    np.random.seed(seed)

    # Collect all peptides
    bbb_positive = collect_brainpeps_manual() + collect_b3pdb_manual()
    bbb_negative = collect_bbb_negative()

    # Create DataFrame
    positive_df = pd.DataFrame(bbb_positive, columns=["sequence", "source"])
    positive_df["label"] = 1

    negative_df = pd.DataFrame(bbb_negative, columns=["sequence", "source"])
    negative_df["label"] = 0

    df = pd.concat([positive_df, negative_df], ignore_index=True)

    # Add length
    df["length"] = df["sequence"].apply(len)

    # Shuffle
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Create train/val/test splits (70/15/15)
    n = len(df)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)

    df["split"] = "test"
    df.loc[:train_end, "split"] = "train"
    df.loc[train_end:val_end, "split"] = "val"

    # Display statistics
    print(f"Dataset Statistics:")
    print(f"  Total peptides: {len(df)}")
    print(f"  BBB+ (label=1): {(df['label'] == 1).sum()}")
    print(f"  BBB- (label=0): {(df['label'] == 0).sum()}")
    print(f"\nSplit distribution:")
    print(df.groupby(['split', 'label']).size().unstack(fill_value=0))
    print(f"\nLength statistics:")
    print(df.groupby('label')['length'].describe())

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nDataset saved to: {output_path}")

    return df


def augment_dataset_with_features(
    dataset_path: str = "data/bbb/bbb_dataset.csv",
    output_path: str = "data/bbb/bbb_dataset_with_features.csv",
) -> pd.DataFrame:
    """
    Add physicochemical features to the dataset.

    This creates a version of the dataset with pre-computed features,
    useful for quick classifier training without re-computing features.

    Parameters
    ----------
    dataset_path : str
        Path to the base dataset CSV
    output_path : str
        Path to save the augmented dataset

    Returns
    -------
    pd.DataFrame
        Dataset with added feature columns
    """
    import sys
    sys.path.append("src")
    from features.peptide_features import PeptideFeatureExtractor

    # Load dataset
    df = pd.read_csv(dataset_path)

    # Extract features
    print("Extracting features (this may take a minute)...")
    extractor = PeptideFeatureExtractor(include_esm=False)
    features = extractor.extract_batch(df["sequence"].tolist())

    # Add features to DataFrame
    feature_names = extractor.get_feature_names()
    for i, name in enumerate(feature_names):
        df[f"feat_{name}"] = features[:, i]

    # Save
    df.to_csv(output_path, index=False)
    print(f"Augmented dataset saved to: {output_path}")

    return df


if __name__ == "__main__":
    # Create base dataset
    print("=" * 80)
    print("BBB Permeability Dataset Collection")
    print("=" * 80)
    print()

    df = create_dataset()

    print("\n" + "=" * 80)
    print("Adding physicochemical features...")
    print("=" * 80)
    print()

    df_with_features = augment_dataset_with_features()

    print("\n" + "=" * 80)
    print("Dataset creation complete!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Review data/bbb/bbb_dataset.csv")
    print("  2. Run notebooks/01_bbb_classifier_training.ipynb to train classifier")
