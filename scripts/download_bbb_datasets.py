"""
Download and process real BBB permeability datasets from published databases.

Sources:
1. BBPpredict (https://i.uestc.edu.cn/BBPpredict/)
   - RFtrainingdataset.fasta
   - Independanttestdataset.fasta

2. B3Pred (https://webs.iiitd.edu.in/raghava/b3pred/)
   - Positive_B3PPs.fasta (BBB+ peptides)
   - Negative_CPPs.fasta (CPPs that don't cross BBB)
   - Negative_Random_Balanced.fasta (Random peptides)
   - Negative_Random.fasta (Random peptides)
"""

import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import re


# Dataset URLs
DATASETS = {
    # BBPpredict datasets
    "bbppredict_train": {
        "url": "https://i.uestc.edu.cn/BBPpredict/data/RFtrainingdataset.fasta",
        "source": "BBPpredict",
        "type": "mixed",  # Contains both positive and negative
    },
    "bbppredict_test": {
        "url": "https://i.uestc.edu.cn/BBPpredict/data/Independanttestdataset.fasta",
        "source": "BBPpredict",
        "type": "mixed",
    },

    # B3Pred datasets
    "b3pred_positive": {
        "url": "https://webs.iiitd.edu.in/raghava/b3pred/Datasets/Positive_B3PPs.fasta",
        "source": "B3Pred",
        "type": "positive",
    },
    "b3pred_negative_cpps": {
        "url": "https://webs.iiitd.edu.in/raghava/b3pred/Datasets/Negative_CPPs.fasta",
        "source": "B3Pred",
        "type": "negative",
    },
    "b3pred_negative_balanced": {
        "url": "https://webs.iiitd.edu.in/raghava/b3pred/Datasets/Negative_Random_Balanced.fasta",
        "source": "B3Pred",
        "type": "negative",
    },
    "b3pred_negative_random": {
        "url": "https://webs.iiitd.edu.in/raghava/b3pred/Datasets/Negative_Random.fasta",
        "source": "B3Pred",
        "type": "negative",
    },
}


def parse_fasta(fasta_content: str) -> List[Tuple[str, str]]:
    """
    Parse FASTA format content.

    Parameters
    ----------
    fasta_content : str
        FASTA file content

    Returns
    -------
    list of tuples
        List of (header, sequence) tuples
    """
    sequences = []
    current_header = None
    current_seq = []

    for line in fasta_content.split('\n'):
        line = line.strip()
        if not line:
            continue

        if line.startswith('>'):
            # Save previous sequence
            if current_header is not None:
                sequences.append((current_header, ''.join(current_seq)))

            # Start new sequence
            current_header = line[1:]  # Remove '>'
            current_seq = []
        else:
            # Add to current sequence
            current_seq.append(line)

    # Save last sequence
    if current_header is not None:
        sequences.append((current_header, ''.join(current_seq)))

    return sequences


def extract_label_from_header(header: str, source: str) -> int:
    """
    Extract BBB label from FASTA header.

    BBPpredict format: ">ID|LABEL|DATASET" where LABEL is 0 or 1
    B3Pred format: Just sequence IDs (use dataset type instead)

    Parameters
    ----------
    header : str
        FASTA header
    source : str
        Dataset source name

    Returns
    -------
    int
        1 for positive, 0 for negative, -1 for unknown
    """
    # BBPpredict uses pipe-separated format: ID|LABEL|DATASET
    if 'BBPpredict' in source:
        parts = header.split('|')
        if len(parts) >= 2:
            try:
                label = int(parts[1].strip())
                return label
            except ValueError:
                pass

    # B3Pred uses text-based labels
    header_lower = header.lower()
    if 'positive' in header_lower or 'pos' in header_lower:
        return 1
    elif 'negative' in header_lower or 'neg' in header_lower:
        return 0

    return -1


def download_dataset(name: str, info: Dict) -> pd.DataFrame:
    """
    Download and parse a single dataset.

    Parameters
    ----------
    name : str
        Dataset name
    info : dict
        Dataset information (url, source, type)

    Returns
    -------
    pd.DataFrame
        Dataset with columns: sequence, label, source, original_id
    """
    print(f"Downloading {name} from {info['source']}...")

    try:
        response = requests.get(info['url'], timeout=30)
        response.raise_for_status()

        # Parse FASTA
        sequences = parse_fasta(response.text)
        print(f"  Found {len(sequences)} sequences")

        # Create DataFrame
        data = []
        for header, seq in sequences:
            # Clean sequence (remove non-amino acid characters)
            seq_clean = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', seq.upper())

            if len(seq_clean) == 0:
                continue

            # Determine label
            if info['type'] == 'positive':
                label = 1
            elif info['type'] == 'negative':
                label = 0
            else:  # mixed
                label = extract_label_from_header(header, info['source'])

            data.append({
                'sequence': seq_clean,
                'label': label,
                'source': info['source'],
                'dataset': name,
                'original_id': header,
            })

        df = pd.DataFrame(data)

        # Filter out unknown labels for mixed datasets
        if info['type'] == 'mixed':
            unknown_count = (df['label'] == -1).sum()
            if unknown_count > 0:
                print(f"  Warning: {unknown_count} sequences with unknown labels (removing)")
                df = df[df['label'] != -1]

        print(f"  Parsed {len(df)} valid sequences")
        print(f"    BBB+: {(df['label'] == 1).sum()}")
        print(f"    BBB-: {(df['label'] == 0).sum()}")

        return df

    except Exception as e:
        print(f"  Error downloading {name}: {e}")
        return pd.DataFrame()


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate sequences, keeping the first occurrence.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with sequences

    Returns
    -------
    pd.DataFrame
        Dataset with duplicates removed
    """
    n_before = len(df)
    df_dedup = df.drop_duplicates(subset=['sequence'], keep='first')
    n_after = len(df_dedup)
    n_removed = n_before - n_after

    print(f"Removed {n_removed} duplicate sequences ({n_removed/n_before*100:.1f}%)")

    return df_dedup


def filter_by_length(df: pd.DataFrame, min_len: int = 5, max_len: int = 50) -> pd.DataFrame:
    """
    Filter peptides by length.

    For BBB-crossing peptides, typical range is 5-50 amino acids.
    Very short (<5) may not be realistic peptides.
    Very long (>50) may be small proteins rather than peptides.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset
    min_len : int, default=5
        Minimum sequence length
    max_len : int, default=50
        Maximum sequence length

    Returns
    -------
    pd.DataFrame
        Filtered dataset
    """
    df['length'] = df['sequence'].apply(len)
    n_before = len(df)

    df_filtered = df[(df['length'] >= min_len) & (df['length'] <= max_len)].copy()
    n_after = len(df_filtered)
    n_removed = n_before - n_after

    print(f"Filtered by length ({min_len}-{max_len} aa): removed {n_removed} sequences")
    print(f"  Length range: {df_filtered['length'].min()}-{df_filtered['length'].max()} aa")
    print(f"  Mean length: {df_filtered['length'].mean():.1f} ± {df_filtered['length'].std():.1f} aa")

    return df_filtered


def create_splits(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15,
                 seed: int = 42) -> pd.DataFrame:
    """
    Create train/val/test splits with stratification by label.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset
    train_frac : float, default=0.70
        Fraction for training
    val_frac : float, default=0.15
        Fraction for validation
    seed : int, default=42
        Random seed

    Returns
    -------
    pd.DataFrame
        Dataset with 'split' column
    """
    from sklearn.model_selection import train_test_split

    df = df.copy()

    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_frac,
        stratify=df['label'],
        random_state=seed
    )

    # Second split: val vs test
    val_size = val_frac / (1 - train_frac)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_size,
        stratify=temp_df['label'],
        random_state=seed
    )

    # Assign splits
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'

    # Combine
    df_final = pd.concat([train_df, val_df, test_df], ignore_index=True)

    print(f"\nSplit distribution:")
    print(df_final.groupby(['split', 'label']).size().unstack(fill_value=0))

    return df_final


def download_and_merge_datasets(
    output_path: str = "data/bbb/bbb_dataset_real.csv",
    min_len: int = 5,
    max_len: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Download all datasets, merge, clean, and save.

    Parameters
    ----------
    output_path : str
        Path to save merged dataset
    min_len : int, default=5
        Minimum peptide length
    max_len : int, default=50
        Maximum peptide length
    seed : int, default=42
        Random seed for splits

    Returns
    -------
    pd.DataFrame
        Final merged dataset
    """
    print("=" * 80)
    print("BBB Dataset Download and Processing")
    print("=" * 80)
    print()

    # Download all datasets
    dfs = []
    for name, info in DATASETS.items():
        df = download_dataset(name, info)
        if len(df) > 0:
            dfs.append(df)
        print()

    if len(dfs) == 0:
        raise ValueError("No datasets were successfully downloaded")

    # Merge all datasets
    print("=" * 80)
    print("Merging datasets...")
    print("=" * 80)
    df_merged = pd.concat(dfs, ignore_index=True)
    print(f"Total sequences (before deduplication): {len(df_merged)}")
    print(f"  BBB+ (label=1): {(df_merged['label'] == 1).sum()}")
    print(f"  BBB- (label=0): {(df_merged['label'] == 0).sum()}")
    print()

    # Remove duplicates
    print("=" * 80)
    print("Removing duplicates...")
    print("=" * 80)
    df_dedup = remove_duplicates(df_merged)
    print(f"After deduplication: {len(df_dedup)} sequences")
    print(f"  BBB+ (label=1): {(df_dedup['label'] == 1).sum()}")
    print(f"  BBB- (label=0): {(df_dedup['label'] == 0).sum()}")
    print()

    # Filter by length
    print("=" * 80)
    print("Filtering by length...")
    print("=" * 80)
    df_filtered = filter_by_length(df_dedup, min_len=min_len, max_len=max_len)
    print(f"After length filter: {len(df_filtered)} sequences")
    print(f"  BBB+ (label=1): {(df_filtered['label'] == 1).sum()}")
    print(f"  BBB- (label=0): {(df_filtered['label'] == 0).sum()}")
    print()

    # Create train/val/test splits
    print("=" * 80)
    print("Creating train/val/test splits...")
    print("=" * 80)
    df_final = create_splits(df_filtered, seed=seed)

    # Display final statistics
    print("\n" + "=" * 80)
    print("Final Dataset Statistics")
    print("=" * 80)
    print(f"Total sequences: {len(df_final)}")
    print(f"  BBB+ (label=1): {(df_final['label'] == 1).sum()}")
    print(f"  BBB- (label=0): {(df_final['label'] == 0).sum()}")
    print()
    print("Source distribution:")
    print(df_final.groupby(['source', 'label']).size().unstack(fill_value=0))
    print()
    print("Length statistics by label:")
    print(df_final.groupby('label')['length'].describe())

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f"\nDataset saved to: {output_path}")

    return df_final


def add_features_to_dataset(
    dataset_path: str = "data/bbb/bbb_dataset_real.csv",
    output_path: str = "data/bbb/bbb_dataset_real_with_features.csv",
) -> pd.DataFrame:
    """
    Add physicochemical features to the dataset.

    Parameters
    ----------
    dataset_path : str
        Path to base dataset
    output_path : str
        Path to save augmented dataset

    Returns
    -------
    pd.DataFrame
        Dataset with features
    """
    import sys
    sys.path.append("src")
    from features.peptide_features import PeptideFeatureExtractor

    print("=" * 80)
    print("Adding physicochemical features...")
    print("=" * 80)

    df = pd.read_csv(dataset_path)
    print(f"Loaded {len(df)} sequences")

    # Extract features (without ESM embeddings for speed)
    print("Extracting features (this may take a few minutes)...")
    extractor = PeptideFeatureExtractor(include_esm=False)

    # Process in batches for progress tracking
    batch_size = 100
    all_features = []

    for i in range(0, len(df), batch_size):
        batch_seqs = df['sequence'].iloc[i:i+batch_size].tolist()
        batch_features = extractor.extract_batch(batch_seqs)
        all_features.append(batch_features)
        print(f"  Processed {min(i+batch_size, len(df))}/{len(df)} sequences", end='\r')

    print()  # New line after progress

    features = np.vstack(all_features)

    # Add features to DataFrame
    feature_names = extractor.get_feature_names()
    for i, name in enumerate(feature_names):
        df[f"feat_{name}"] = features[:, i]

    # Save
    df.to_csv(output_path, index=False)
    print(f"Augmented dataset saved to: {output_path}")
    print(f"  Total columns: {len(df.columns)}")
    print(f"  Feature columns: {len(feature_names)}")

    return df


if __name__ == "__main__":
    # Download and process datasets
    df = download_and_merge_datasets(
        output_path="data/bbb/bbb_dataset_real.csv",
        min_len=5,
        max_len=50,
        seed=42,
    )

    print("\n" + "=" * 80)
    print("Adding features...")
    print("=" * 80)

    df_with_features = add_features_to_dataset(
        dataset_path="data/bbb/bbb_dataset_real.csv",
        output_path="data/bbb/bbb_dataset_real_with_features.csv",
    )

    print("\n" + "=" * 80)
    print("Dataset Download Complete!")
    print("=" * 80)
    print()
    print("Files created:")
    print("  1. data/bbb/bbb_dataset_real.csv")
    print("  2. data/bbb/bbb_dataset_real_with_features.csv")
    print()
    print("Next steps:")
    print("  1. Review the datasets")
    print("  2. Update notebooks/01_bbb_classifier_training.ipynb to use the real dataset")
    print("  3. Train the BBB classifier")
