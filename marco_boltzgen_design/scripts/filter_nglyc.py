#!/usr/bin/env python3
"""
filter_nglyc.py — Remove designs with N-glycosylation sequons from BoltzGen metrics.

BoltzProt-1 protocol (Technical Report Appendix E.1): excludes all 32 NXS/T
sequons from binder sequences to avoid glycan heterogeneity during expression.
Motif list matches the Boltz API `excluded_sequence_motifs` exactly.

Usage:
    python scripts/filter_nglyc.py --metrics results/all_designs_metrics.csv
    python scripts/filter_nglyc.py --metrics results/all_designs_metrics.csv --dry_run

Output:
    Writes filtered CSV back to --out (default: overwrites input).
    Prints summary of removed designs and remaining designs.
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# All 32 N-glycosylation sequons (N-X-S/T where X != P)
# Matches Boltz API excluded_sequence_motifs exactly
NGLYC_MOTIFS_32 = [
    "NAS", "NAT", "NCS", "NCT", "NDS", "NDT",
    "NES", "NET", "NFS", "NFT", "NGS", "NGT",
    "NHS", "NHT", "NIS", "NIT", "NKS", "NKT",
    "NLS", "NLT", "NMS", "NMT", "NNS", "NNT",
    "NQS", "NQT", "NRS", "NRT", "NSS", "SST",
    "NTS", "NTT", "NVS", "NVT", "NWS", "NWT",
    "NYS", "NYT",
]

# Combined regex: N[^P][ST] (native regex equivalent)
NGLYC_PATTERN = re.compile(r"N[^P][ST]")


def infer_sequence(row) -> str:
    for k in ("designed_sequence", "binder_sequence", "sequence", "seq"):
        if k in row and isinstance(row[k], str):
            return row[k]
    return ""


def has_nglyc(seq: str) -> bool:
    return bool(NGLYC_PATTERN.search(seq or ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", required=True, help="BoltzGen all_designs_metrics.csv")
    ap.add_argument("--out", help="Output CSV (default: overwrites --metrics)")
    ap.add_argument("--dry_run", action="store_true", help="Print stats but do not write")
    ap.add_argument(
        "--method",
        default="regex32",
        choices=["regex32", "simple"],
        help="'regex32' uses the 32-motif list (Boltz API match); 'simple' uses N[^P][ST] (broader)",
    )
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else Path(args.metrics)

    df = pd.read_csv(args.metrics)
    seq_col_exists = "binder_sequence" in df.columns or "designed_sequence" in df.columns
    df["_seq"] = df.apply(infer_sequence, axis=1)

    # Count before
    n_before = len(df)

    # Identify N-glyc designs
    if args.method == "simple":
        df["_has_nglyc"] = df["_seq"].apply(lambda s: bool(NGLYC_PATTERN.search(s or "")))
    else:
        df["_has_nglyc"] = df["_seq"].apply(lambda s: any(m in (s or "") for m in NGLYC_MOTIFS_32))

    n_nglyc = df["_has_nglyc"].sum()
    pct_nglyc = 100 * n_nglyc / n_before if n_before else 0

    # Per-motif breakdown
    if args.method == "regex32":
        motif_counts = {}
        for motif in NGLYC_MOTIFS_32:
            count = df["_seq"].str.contains(motif, na=False).sum()
            if count > 0:
                motif_counts[motif] = count
    else:
        motif_counts = {}

    # Keep non-NGLYC designs
    df_clean = df[~df["_has_nglyc"]].copy()
    df_clean.drop(columns=["_seq", "_has_nglyc"], inplace=True, errors="ignore")
    n_after = len(df_clean)

    print(f"=== N-glycosylation filter ===")
    print(f"  Input:              {args.metrics}")
    print(f"  Designs before:     {n_before}")
    print(f"  N-glyc designs:      {n_nglyc} ({pct_nglyc:.1f}%)")
    if motif_counts:
        print(f"  Per-motif counts:")
        for motif, count in sorted(motif_counts.items(), key=lambda x: -x[1]):
            print(f"    {motif}: {count}")
    print(f"  Designs after:      {n_after}")

    if args.dry_run:
        print("  [dry_run] — no file written")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(out_path, index=False)
    print(f"  Written:             {out_path}")
    print(f"\n  NOTE: Run 'python scripts/rank_designs.py --metrics {out_path} ...' to re-rank after filtering.")


if __name__ == "__main__":
    main()