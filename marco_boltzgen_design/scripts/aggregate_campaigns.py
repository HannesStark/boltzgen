#!/usr/bin/env python3
"""Aggregate metrics across multiple BoltzGen design campaigns and re-rank.

Usage:
    python scripts/aggregate_campaigns.py \
        --root runs/ \
        --out results/aggregated_metrics.csv

Output columns: campaign, design_id, <all columns from all_designs_metrics.csv>
Duplicates are deduplicated by binder_sequence (best score wins).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def infer_sequence(row):
    """Mirror rank_designs.py logic for consistency."""
    for k in ["designed_sequence", "binder_sequence", "sequence", "seq"]:
        if k in row and isinstance(row[k], str):
            return row[k]
    return ""


def find_metrics_files(root: Path):
    """Recursively find all BoltzGen all_designs_metrics.csv under root."""
    pattern = "**/all_designs_metrics.csv"
    return sorted(root.glob(pattern))


def load_with_campaign(path: Path, campaign_label: str) -> pd.DataFrame:
    """Load a single metrics CSV and attach a campaign label."""
    df = pd.read_csv(path)
    df["campaign"] = campaign_label
    return df


def deduplicate(df: pd.DataFrame, score_col: str = "final_score") -> pd.DataFrame:
    """Deduplicate by binder_sequence, keeping the row with the highest score."""
    df = df.copy()
    df["_binder_seq"] = df.apply(infer_sequence, axis=1)

    # Rows with sequences — deduplicate
    with_seq = df[df["_binder_seq"] != ""].copy()
    without_seq = df[df["_binder_seq"] == ""].copy()

    if score_col in with_seq.columns:
        # Keep highest-scoring entry per sequence
        with_seq_dedup = (
            with_seq.sort_values(score_col, ascending=False)
            .groupby("_binder_seq", sort=False)
            .first()
            .reset_index()
        )
    else:
        # No score column — keep first seen
        with_seq_dedup = (
            with_seq.groupby("_binder_seq", sort=False)
            .first()
            .reset_index()
        )

    result = pd.concat([with_seq_dedup, without_seq], ignore_index=True)
    result.drop(columns=["_binder_seq"], inplace=True, errors="ignore")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("runs"),
        help="Root directory containing campaign subdirectories (default: runs/)",
    )
    ap.add_argument(
        "--pattern",
        default="**/all_designs_metrics.csv",
        help="Glob pattern relative to root (default: **/all_designs_metrics.csv)",
    )
    ap.add_argument(
        "--score-col",
        default="final_score",
        help="Column to use for deduplication tie-breaking (default: final_score)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("results/aggregated_metrics.csv"),
        help="Output CSV path (default: results/aggregated_metrics.csv)",
    )
    args = ap.parse_args()

    metrics_files = find_metrics_files(args.root / args.pattern.replace("**/", ""))
    if not metrics_files:
        print(f"ERROR: No metrics files found under {args.root} with pattern '{args.pattern}'", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(metrics_files)} metrics files:")
    for f in metrics_files:
        print(f"  {f}")

    dfs = []
    for path in metrics_files:
        # Derive campaign label from parent directory name
        campaign = path.parent.name
        df = load_with_campaign(path, campaign)
        n = len(df)
        dfs.append(df)
        print(f"  {campaign}: {n} designs")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal before deduplication: {len(combined)} rows")

    # Deduplicate
    if "final_score" not in combined.columns and args.score_col not in combined.columns:
        print("NOTE: No score column found — deduplicating by first-seen per sequence.")
    deduped = deduplicate(combined, score_col=args.score_col)
    print(f"Total after deduplication:  {len(deduped)} rows")

    # Re-rank by final_score if present
    if args.score_col in deduped.columns:
        ranked = deduped.sort_values(args.score_col, ascending=False).reset_index(drop=True)
    else:
        ranked = deduped.reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}  ({len(ranked)} unique designs across {len(dfs)} campaigns)")


if __name__ == "__main__":
    main()