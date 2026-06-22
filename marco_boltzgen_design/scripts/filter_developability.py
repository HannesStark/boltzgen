#!/usr/bin/env python3
"""
filter_developability.py — Remove designs with N-glycosylation sequons and/or
proline-in-CDR3 from BoltzGen metrics, before ranking.

Filters:
  N-glyc  — NXS/T sequons (32-motif list, Boltz API match) → glycan heterogeneity
  Pro CDR3 — proline in the central CDR3 region (approx. middle 30% of sequence)
             → β-sheet disruption, Tm loss

BoltzProt-1 protocol: excludes both at generation time for highest confirmed-binder
rate. When running post-generation (e.g. after a partial re-run), this script
applies them as hard gates before the ranking step.

Usage:
    # Both filters (recommended):
    python scripts/filter_developability.py --metrics results/all_designs_metrics.csv \
        --filter_nglyc --filter_proline --out results/all_designs_metrics.csv

    # N-glyc only (matches legacy filter_nglyc.py behaviour):
    python scripts/filter_developability.py --metrics results/all_designs_metrics.csv \
        --filter_nglyc --out results/all_designs_metrics.csv

    # Dry run (see counts without writing):
    python scripts/filter_developability.py --metrics results/all_designs_metrics.csv \
        --filter_nglyc --filter_proline --dry_run
"""
from __future__ import annotations
import argparse
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


def infer_sequence(row) -> str:
    for k in ("designed_sequence", "binder_sequence", "sequence", "seq"):
        if k in row and isinstance(row[k], str):
            return row[k]
    return ""


def has_nglyc(seq: str) -> bool:
    """N[^P][ST] — native regex equivalent to the 32-motif list."""
    if not seq:
        return False
    for i in range(len(seq) - 2):
        if seq[i] == "N":
            if seq[i + 1] != "P":
                if seq[i + 2] in "ST":
                    return True
    return False


def proline_in_cdr3(seq: str) -> bool:
    """Proline in CDR3 region (last ~18% of VHH sequence).

    For a typical ~113-aa VHH, CDR3 occupies the C-terminal ~18 residues
    (~Kabat positions 95–113). FR3 contains conserved prolines (e.g. PGK,
    PW) so the old 1/3–2/3 heuristic falsely flags framework as CDR3.
    Checking the last 18% correctly captures CDR3 for sequences ≥ 60 aa.
    """
    if not seq or len(seq) < 60:
        return False
    cdr3_region = seq[int(len(seq) * 0.82):]
    return "P" in cdr3_region


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--metrics", required=True,
                    help="BoltzGen all_designs_metrics.csv")
    ap.add_argument("--out",
                    help="Output CSV (default: overwrites --metrics)")
    ap.add_argument("--filter_nglyc", action="store_true",
                    help="Remove designs with N-glycosylation sequons (N[^P][ST])")
    ap.add_argument("--filter_proline", action="store_true",
                    help="Remove designs with proline in CDR3 region")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print stats but do not write")
    args = ap.parse_args()

    if not (args.filter_nglyc or args.filter_proline):
        print("ERROR: at least one of --filter_nglyc or --filter_proline is required.")
        sys.exit(1)

    out_path = Path(args.out) if args.out else Path(args.metrics)

    df = pd.read_csv(args.metrics)
    df["_seq"] = df.apply(infer_sequence, axis=1)

    n_before = len(df)

    # N-glyc filter
    if args.filter_nglyc:
        df["_has_nglyc"] = df["_seq"].apply(has_nglyc)
        # Per-motif breakdown for reporting
        motif_counts = {}
        for motif in NGLYC_MOTIFS_32:
            count = df["_seq"].str.contains(motif, na=False).sum()
            if count > 0:
                motif_counts[motif] = count
    else:
        df["_has_nglyc"] = False
        motif_counts = {}

    # Proline-in-CDR3 filter
    if args.filter_proline:
        df["_has_proline"] = df["_seq"].apply(proline_in_cdr3)
    else:
        df["_has_proline"] = False

    n_nglyc = int(df["_has_nglyc"].sum())
    n_proline = int(df["_has_proline"].sum())

    # Combined mask
    df["_filtered"] = df["_has_nglyc"] | df["_has_proline"]
    n_removed = int(df["_filtered"].sum())

    # Keep clean designs
    df_clean = df[~df["_filtered"]].copy()
    df_clean.drop(columns=["_seq", "_has_nglyc", "_has_proline", "_filtered"],
                  inplace=True, errors="ignore")
    n_after = len(df_clean)

    # ── Reporting ────────────────────────────────────────────────────────────
    print(f"=== Developability filter ===")
    print(f"  Input:              {args.metrics}")
    print(f"  Designs before:     {n_before}")
    if args.filter_nglyc:
        print(f"  N-glyc removed:       {n_nglyc} ({100*n_nglyc/n_before:.1f}%)")
        if motif_counts:
            print(f"  Per-motif counts:")
            for motif, count in sorted(motif_counts.items(), key=lambda x: -x[1]):
                print(f"    {motif}: {count}")
    if args.filter_proline:
        print(f"  Proline-CDR3 removed: {n_proline} ({100*n_proline/n_before:.1f}%)")
    print(f"  Designs after:      {n_after}")
    print(f"  Total removed:       {n_removed} ({100*n_removed/n_before:.1f}%)")

    if args.dry_run:
        print("  [dry_run] — no file written")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(out_path, index=False)
    print(f"  Written:             {out_path}")
    print(f"\n  NOTE: Run 'python scripts/rank_designs.py --metrics {out_path} ...' to re-rank after filtering.")


if __name__ == "__main__":
    main()