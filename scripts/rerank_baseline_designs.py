#!/usr/bin/env python3
"""
Post-hoc Ranking of Baseline Peptide Designs

Phase 3, Task 10: Re-rank baseline peptides using BBB + TAU composite scoring

Usage:
    python scripts/rerank_baseline_designs.py \
        --input workbench/baseline_60k/final_ranked_designs/final_*_designs \
        --output workbench/reranked \
        --top_k 100 \
        --bbb_weight 0.6 \
        --tau_weight 0.4
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scoring.composite_scorer import CompositeScorer


def parse_cif_sequence(cif_file: Path) -> str:
    """
    Extract sequence from mmCIF file.

    Parameters
    ----------
    cif_file : Path
        Path to CIF file

    Returns
    -------
    str
        Extracted sequence
    """
    sequence = []

    with open(cif_file, 'r') as f:
        in_atom_site = False

        for line in f:
            if line.startswith('_atom_site.'):
                in_atom_site = True
                continue

            if in_atom_site and line.startswith('#'):
                break

            if in_atom_site and line.strip():
                # Parse atom_site loop
                parts = line.split()
                if len(parts) >= 5:
                    # Typically: group_PDB, id, label_comp_id, label_asym_id, ...
                    res_name = parts[2] if len(parts) > 2 else None

                    # Convert 3-letter to 1-letter code
                    if res_name and len(res_name) == 3:
                        aa = three_to_one(res_name)
                        if aa and aa not in sequence[-1:]:  # Avoid duplicates
                            sequence.append(aa)

    return ''.join(sequence)


def three_to_one(three_letter: str) -> str:
    """Convert 3-letter amino acid code to 1-letter."""
    conversion = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
        'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    }
    return conversion.get(three_letter.upper(), '')


def extract_sequences_from_boltzgen_output(output_dir: Path) -> pd.DataFrame:
    """
    Extract sequences from BoltzGen output directory.

    Parameters
    ----------
    output_dir : Path
        BoltzGen output directory (e.g., workbench/baseline_60k)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: design_id, sequence, source_file
    """
    # Look for CIF files in final_ranked_designs
    final_dir = output_dir / "final_ranked_designs"

    if not final_dir.exists():
        raise FileNotFoundError(f"Final designs directory not found: {final_dir}")

    # Find all design subdirectories
    design_dirs = list(final_dir.glob("final_*_designs"))

    if not design_dirs:
        raise FileNotFoundError(f"No design directories found in {final_dir}")

    print(f"Found {len(design_dirs)} design directories")

    sequences = []

    for design_dir in design_dirs:
        cif_files = list(design_dir.glob("*.cif"))

        for cif_file in tqdm(cif_files, desc=f"Processing {design_dir.name}"):
            try:
                seq = parse_cif_sequence(cif_file)

                if seq:
                    sequences.append({
                        'design_id': cif_file.stem,
                        'sequence': seq,
                        'source_file': str(cif_file.relative_to(output_dir)),
                        'length': len(seq),
                    })
            except Exception as e:
                print(f"Warning: Failed to parse {cif_file.name}: {e}")

    df = pd.DataFrame(sequences)

    print(f"\\nExtracted {len(df)} sequences")
    print(f"Length distribution: {df['length'].min()}-{df['length'].max()} aa")

    return df


def score_and_rank(
    sequences_df: pd.DataFrame,
    scorer: CompositeScorer,
    top_k: int = 100,
) -> pd.DataFrame:
    """
    Score sequences and rank by composite score.

    Parameters
    ----------
    sequences_df : pd.DataFrame
        DataFrame with sequence column
    scorer : CompositeScorer
        Composite scorer instance
    top_k : int
        Number of top designs to return

    Returns
    -------
    pd.DataFrame
        Ranked designs with scores
    """
    print("\\nScoring sequences...")

    scores = []

    for _, row in tqdm(sequences_df.iterrows(), total=len(sequences_df)):
        try:
            score_dict = scorer.score(row['sequence'])
            score_dict['design_id'] = row['design_id']
            score_dict['source_file'] = row['source_file']
            score_dict['length'] = row['length']
            scores.append(score_dict)
        except Exception as e:
            print(f"Warning: Failed to score {row['design_id']}: {e}")

    # Create DataFrame and sort
    df = pd.DataFrame(scores)
    df = df.sort_values('composite', ascending=False)

    print(f"\\nTop scores:")
    print(f"  Composite: {df['composite'].max():.3f} (mean: {df['composite'].mean():.3f})")
    print(f"  BBB prob:  {df['bbb_prob'].max():.3f} (mean: {df['bbb_prob'].mean():.3f})")
    print(f"  TAU score: {df['tau_score'].max():.3f} (mean: {df['tau_score'].mean():.3f})")

    # Select top-k
    if top_k:
        df_top = df.head(top_k)
        print(f"\\nSelected top-{top_k} designs")
    else:
        df_top = df

    return df_top


def save_results(
    df: pd.DataFrame,
    output_dir: Path,
    save_fasta: bool = True,
):
    """
    Save reranked results.

    Parameters
    ----------
    df : pd.DataFrame
        Scored and ranked designs
    output_dir : Path
        Output directory
    save_fasta : bool
        Whether to save FASTA file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    csv_file = output_dir / "reranked_designs.csv"
    df.to_csv(csv_file, index=False)
    print(f"\\nSaved results to {csv_file}")

    # Save FASTA
    if save_fasta:
        fasta_file = output_dir / "reranked_designs.fasta"
        with open(fasta_file, 'w') as f:
            for _, row in df.iterrows():
                f.write(f">{row['design_id']} | composite={row['composite']:.3f} | ")
                f.write(f"BBB={row['bbb_prob']:.3f} | TAU={row['tau_score']:.3f}\\n")
                f.write(f"{row['sequence']}\\n")
        print(f"Saved FASTA to {fasta_file}")

    # Save summary statistics
    summary = {
        'total_designs': len(df),
        'score_statistics': {
            'composite': {
                'min': float(df['composite'].min()),
                'max': float(df['composite'].max()),
                'mean': float(df['composite'].mean()),
                'std': float(df['composite'].std()),
            },
            'bbb_prob': {
                'min': float(df['bbb_prob'].min()),
                'max': float(df['bbb_prob'].max()),
                'mean': float(df['bbb_prob'].mean()),
                'std': float(df['bbb_prob'].std()),
            },
            'tau_score': {
                'min': float(df['tau_score'].min()),
                'max': float(df['tau_score'].max()),
                'mean': float(df['tau_score'].mean()),
                'std': float(df['tau_score'].std()),
            },
        },
        'length_distribution': {
            'min': int(df['length'].min()),
            'max': int(df['length'].max()),
            'mean': float(df['length'].mean()),
        },
    }

    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Re-rank baseline peptide designs using BBB + TAU scoring"
    )
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='BoltzGen output directory (e.g., workbench/baseline_60k)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('workbench/reranked'),
        help='Output directory for reranked results'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=100,
        help='Number of top designs to save (default: 100, 0 for all)'
    )
    parser.add_argument(
        '--bbb_weight',
        type=float,
        default=0.6,
        help='Weight for BBB score (default: 0.6)'
    )
    parser.add_argument(
        '--tau_weight',
        type=float,
        default=0.4,
        help='Weight for TAU score (default: 0.4)'
    )
    parser.add_argument(
        '--bbb_model',
        type=str,
        default='models/bbb_classifier.pt',
        help='Path to BBB classifier checkpoint'
    )

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        print(f"Error: Input directory not found: {args.input}")
        sys.exit(1)

    print("=" * 80)
    print("POST-HOC RANKING OF BASELINE DESIGNS")
    print("=" * 80)
    print(f"\\nInput:       {args.input}")
    print(f"Output:      {args.output}")
    print(f"Top-K:       {args.top_k}")
    print(f"BBB weight:  {args.bbb_weight}")
    print(f"TAU weight:  {args.tau_weight}")

    # Load scorer
    print("\\nLoading composite scorer...")
    try:
        scorer = CompositeScorer.from_checkpoints(
            bbb_model_path=args.bbb_model,
            bbb_weight=args.bbb_weight,
            tau_weight=args.tau_weight,
        )
    except Exception as e:
        print(f"Error loading scorer: {e}")
        print("\\nMake sure BBB classifier and TAU data are available:")
        print(f"  - {args.bbb_model}")
        print(f"  - data/tau/tau_sequence_info.txt")
        print(f"  - data/tau/tau_entropy.npy")
        print(f"  - data/tau/tau_target_regions.json")
        sys.exit(1)

    # Extract sequences
    print("\\nExtracting sequences from BoltzGen output...")
    sequences_df = extract_sequences_from_boltzgen_output(args.input)

    # Score and rank
    ranked_df = score_and_rank(sequences_df, scorer, args.top_k)

    # Save results
    save_results(ranked_df, args.output)

    print("\\n" + "=" * 80)
    print("RANKING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
