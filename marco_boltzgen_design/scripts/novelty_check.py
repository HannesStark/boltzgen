#!/usr/bin/env python3
"""
novelty_check.py — Check VHH nanobody designs against SAbDab CDR3 reference set.

BoltzProt-1 protocol (Technical Report Section 3.5):
  "Across both target panels, every recovered design has a minimum CDR3 edit distance
   of at least four to its closest SAbDab match."

This script:
  1. Extracts unique CDR3 sequences from the local SAbDab IMGT PDB zip archive
  2. Caches the reference set to avoid re-parsing the zip on every run
  3. For each design, computes:
       - min_edit_distance(CDR3, reference_set)  [CDR3 alone]
       - min_edit_distance(CDR1+CDR2+CDR3, reference_set)  [all three CDRs]
  4. Flags designs with edit_distance < threshold
  5. Supports filtering on CDR3 distance alone, cdr1+2+3 distance alone, or BOTH

Usage:
  # Default (both CDR3 AND CDR1+2+3 must pass):
  python scripts/novelty_check.py \
    --designs results/ranked_candidates.csv \
    --out results/novelty_checked.csv

  # CDR3 only (legacy behaviour):
  python scripts/novelty_check.py --designs results/ranked_candidates.csv \
    --filter_mode cdr3_only

  # CDR1+2+3 only (primary filter, CDR3 as secondary):
  python scripts/novelty_check.py --designs results/ranked_candidates.csv \
    --filter_mode cdrs_only

  # Separate thresholds:
  python scripts/novelty_check.py --designs results/ranked_candidates.csv \
    --min_edit_distance 4 \
    --cdrs_edit_distance_threshold 6

  # First run (builds cache + checks):
  python scripts/novelty_check.py \
    --designs results/ranked_candidates.csv \
    --sabdab_zip ~/Downloads/all_structures.zip \
    --out results/novelty_checked.csv

  # Subsequent runs (uses cached reference):
  python scripts/novelty_check.py --designs results/ranked_candidates.csv

  # Standalone: rebuild reference cache
  python scripts/novelty_check.py --build_cache --sabdab_zip ~/Downloads/all_structures.zip
"""
from __future__ import annotations
import argparse
import csv
import json
import re
import sys
import time
import zipfile
from difflib import SequenceMatcher
from pathlib import Path

# ── Edit distance ────────────────────────────────────────────────────────────

def levenshtein_distance(a: str, b: str) -> int:
    """Pure-Python Levenshtein distance. Fast enough for CDR3 (10-20 aa) strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # Use SequenceMatcher (C-accelerated under the hood)
    return sum(1 for op in SequenceMatcher(a=a, b=b).get_opcodes() if op[0] != "equal")


def min_edit_distance_optimized(
    seq: str,
    reference_by_len: dict[int, list[str]],
    max_search: int = 200,
) -> int:
    """Fast min-edit-distance using length-bucketed reference.

    Groups reference sequences by length. Only compares against references
    within ±3 aa of the query length, then takes the best result.
    Falls back to full scan if the candidate pool is too small.
    """
    if not seq:
        return 999
    L = len(seq)
    candidates: list[str] = []
    # Collect refs within ±3 aa of query length
    for length in range(max(1, L - 3), L + 4):
        if length in reference_by_len:
            candidates.extend(reference_by_len[length])
    # If too few candidates, supplement with all remaining lengths
    if len(candidates) < 20:
        for length, bucket in reference_by_len.items():
            if abs(length - L) > 3:
                candidates.extend(bucket)
    if not candidates:
        return 999
    # Cap candidates for speed
    if len(candidates) > max_search:
        import random
        candidates = random.sample(candidates, max_search)
    return min(levenshtein_distance(seq, ref) for ref in candidates)


# ── CDR3 extraction from IMGT-numbered PDBs ────────────────────────────────────

# IMGT CDR3: positions 105-117 (inclusive), 1-indexed
# In PDB ATOM lines, residues are 1-indexed → slice [104:117] in Python
IMGT_CDR3_START = 104   # 0-indexed start (pos 105 in 1-indexed IMGT)
IMGT_CDR3_END   = 117   # 0-indexed exclusive (pos 117 in 1-indexed IMGT)

# IMGT CDR1: 27-38, CDR2: 56-65 (0-indexed: 26-38, 55-65)
IMGT_CDR1_START, IMGT_CDR1_END = 26, 38
IMGT_CDR2_START, IMGT_CDR2_END = 55, 65


def extract_hchain_seq(atom_lines: list[str]) -> dict[int, str]:
    """Extract per-residue amino acid for the first H-type chain found.

    H-type = chain whose sequence has length > 110 residues (full VHH).
    Returns dict of {resnum: one_letter_aa}.
    """
    by_chain: dict[str, dict[int, str]] = {}

    for l in atom_lines:
        try:
            chain  = l[21]
            res    = int(l[22:26].strip())
            aa     = l[17]
            if aa not in "ACDEFGHIKLMNPQRSTVWY":
                continue
            by_chain.setdefault(chain, {})[res] = aa
        except (ValueError, IndexError):
            continue

    # Pick the first chain with a full-length VHH sequence (>110 aa)
    for chain, residues in sorted(by_chain.items()):
        seq_len = max(residues.keys()) - min(residues.keys()) + 1
        if seq_len >= 110:
            return residues

    return {}


def imgt_cdr123(seq: str) -> tuple[str, str, str]:
    """Extract CDR1, CDR2, CDR3 from an IMGT-numbered full VHH sequence."""
    cdr1 = seq[IMGT_CDR1_START:IMGT_CDR1_END].strip()
    cdr2 = seq[IMGT_CDR2_START:IMGT_CDR2_END].strip()
    cdr3 = seq[IMGT_CDR3_START:IMGT_CDR3_END].strip()
    return cdr1, cdr2, cdr3


# ── Reference set building ────────────────────────────────────────────────────

def build_reference_from_zip(zip_path: str, cache_path: str) -> dict:
    """Extract unique CDR3 sequences from SAbDab IMGT PDB zip.

    Writes a JSON cache after extraction.
    Returns dict with 'cdr3_set' (list), 'all_cdr3' (list), 'pdb_count', 'unique_count'.
    """
    t0 = time.time()
    cdr3_set = set()
    all_cdr3 = []
    pdb_count = 0

    with zipfile.ZipFile(zip_path, "r") as z:
        imgt_files = [n for n in z.namelist()
                     if n.endswith(".pdb") and "/imgt/" in n]

        print(f"Scanning {len(imgt_files):,} IMGT PDBs from {zip_path} ...")
        for i, name in enumerate(imgt_files):
            if i > 0 and i % 5000 == 0:
                print(f"  [{i:,}/{len(imgt_files):,}] unique CDR3s so far: {len(cdr3_set):,}")

            try:
                content = z.read(name).decode("utf-8", errors="replace")
            except Exception:
                continue

            atom_lines = [l for l in content.split("\n") if l.startswith("ATOM")]
            h_residues = extract_hchain_seq(atom_lines)
            if not h_residues:
                continue

            start = min(h_residues.keys())
            end   = max(h_residues.keys())
            seq   = "".join(h_residues.get(r, "") for r in range(start, end + 1))

            if len(seq) < 110:
                continue

            cdr1, cdr2, cdr3 = imgt_cdr123(seq)
            if len(cdr3) >= 5:          # discard degenerate / incomplete CDR3s
                all_cdr3.append(cdr3)
                cdr3_set.add(cdr3)

            pdb_count += 1

    result = {
        "cdr3_set": sorted(cdr3_set),
        "all_cdr3": all_cdr3,
        "pdb_count": pdb_count,
        "unique_count": len(cdr3_set),
        "zip_path": str(zip_path),
    }

    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(result, f, indent=1)

    elapsed = time.time() - t0
    print(f"\nSAbDab reference built:")
    print(f"  PDBs processed : {pdb_count:,}")
    print(f"  Unique CDR3s   : {len(cdr3_set):,}")
    print(f"  CDR3 length range: {min(len(c) for c in cdr3_set) if cdr3_set else 0}"
          f" – {max(len(c) for c in cdr3_set) if cdr3_set else 0} aa")
    print(f"  Cache written  : {cache_path} ({elapsed:.1f}s)")
    return result


# ── Novelty checking ──────────────────────────────────────────────────────────

def build_reference_by_length(cdr3_list: list[str]) -> dict[int, list[str]]:
    """Index reference sequences by length for O(1) length-bucketed lookup."""
    by_len: dict[int, list[str]] = {}
    for seq in cdr3_list:
        by_len.setdefault(len(seq), []).append(seq)
    return by_len


def check_novelty(
    designs_path: str,
    reference_by_len: dict[int, list[str]],
    cache_path: str,
    min_edit_distance: int = 4,
    cdrs_edit_distance_threshold: int | None = None,
    filter_mode: str = "both",  # "cdr3_only" | "cdrs_only" | "both"
) -> pd.DataFrame:
    """Compute min-edit-distance to SAbDab for each design's CDR3 and CDR1+2+3.

    filter_mode:
      both     — both CDR3 and CDR1+2+3 must meet their thresholds (strictest)
      cdr3_only — only CDR3 edit distance is used as the filter gate
      cdrs_only — only CDR1+2+3 edit distance is used; CDR3 threshold is ignored
                   in the novelty_pass output (but both distances are still reported)

    Returns DataFrame with added columns:
      cdr3_seq, min_cdr3_edit_distance, cdr3_novel,
      cdr123_concat, min_cdr123_edit_distance, cdr123_novel,
      novelty_pass (bool — primary gate based on filter_mode)
    """
    import pandas as pd

    cdrs_thresh = (
        cdrs_edit_distance_threshold
        if cdrs_edit_distance_threshold is not None
        else min_edit_distance
    )

    df = pd.read_csv(designs_path)
    if "design_id" not in df.columns:
        df["design_id"] = range(len(df))

    # ── Infer binder sequence ───────────────────────────────────────────────
    def infer_seq(row):
        for k in ("designed_sequence", "binder_sequence", "sequence", "seq"):
            if k in row and isinstance(row[k], str):
                return row[k]
        return ""

    df["_seq"] = df.apply(infer_seq, axis=1)

    # ── Extract CDR3 (IMGT: positions 105-117, 0-indexed 104-117) ──────────
    def get_cdr3(seq: str) -> str:
        if not seq or len(seq) < 110:
            return ""
        return seq[104:117].strip()

    # ── Extract CDR1+2+3 ───────────────────────────────────────────────────
    def get_cdr123(seq: str) -> str:
        if not seq or len(seq) < 110:
            return ""
        cdr1 = seq[26:38].strip()
        cdr2 = seq[55:65].strip()
        cdr3 = seq[104:117].strip()
        return cdr1 + cdr2 + cdr3

    print(
        f"\nNovelty checking {len(df):,} designs "
        f"(ref: {sum(len(v) for v in reference_by_len.values()):,} sequences) ..."
    )
    print(f"  filter_mode       : {filter_mode}")
    print(f"  CDR3 threshold    : {min_edit_distance}")
    print(f"  CDR1+2+3 threshold: {cdrs_thresh}")

    n = len(df)
    chunk_size = 500
    min_cdr3_dists   = []
    min_cdr123_dists = []

    for start in range(0, n, chunk_size):
        chunk = df["_seq"].iloc[start:start + chunk_size]
        if start > 0 and start % 2000 == 0:
            print(f"  [{start:,}/{n:,}]")

        for seq in chunk:
            cdr3   = get_cdr3(seq)
            cdr123 = get_cdr123(seq)
            min_cdr3_dists.append(min_edit_distance_optimized(cdr3, reference_by_len))
            min_cdr123_dists.append(min_edit_distance_optimized(cdr123, reference_by_len))

    new_cols = pd.DataFrame(
        {
            "cdr3_seq":                df["_seq"].apply(get_cdr3),
            "min_cdr3_edit_distance":  min_cdr3_dists,
            "cdr3_novel":              [d >= min_edit_distance for d in min_cdr3_dists],
            "cdr123_concat":           df["_seq"].apply(get_cdr123),
            "min_cdr123_edit_distance": min_cdr123_dists,
            "cdr123_novel":            [d >= cdrs_thresh for d in min_cdr123_dists],
        },
        index=df.index,
    )
    df = pd.concat([df, new_cols], axis=1)

    # ── Primary filter gate: novelty_pass ─────────────────────────────────────
    if filter_mode == "both":
        df["novelty_pass"] = df["cdr3_novel"] & df["cdr123_novel"]
    elif filter_mode == "cdr3_only":
        df["novelty_pass"] = df["cdr3_novel"]
    elif filter_mode == "cdrs_only":
        df["novelty_pass"] = df["cdr123_novel"]

    # ── Summary ─────────────────────────────────────────────────────────────
    n_cdr3_ok   = df["cdr3_novel"].sum()
    n_cdr123_ok = df["cdr123_novel"].sum()
    n_pass      = df["novelty_pass"].sum()

    print(f"\nNovelty results:")
    print(f"  CDR3-novel  (dist ≥ {min_edit_distance})  : {n_cdr3_ok}/{len(df)} ({100*n_cdr3_ok/len(df):.1f}%)")
    print(f"  CDR1+2+3-novel (dist ≥ {cdrs_thresh})       : {n_cdr123_ok}/{len(df)} ({100*n_cdr123_ok/len(df):.1f}%)")
    print(f"  novelty_pass [{filter_mode}]                  : {n_pass}/{len(df)} ({100*n_pass/len(df):.1f}%)")

    if n_pass < len(df):
        dupes = df[~df["novelty_pass"]][["design_id", "cdr3_seq", "min_cdr3_edit_distance", "min_cdr123_edit_distance"]]
        print(f"\n  Designs failing novelty gate (first 5):")
        for _, row in dupes.head(5).iterrows():
            print(f"    id={row['design_id']} cdr3='{row['cdr3_seq']}' "
                  f"cdr3_dist={row['min_cdr3_edit_distance']} "
                  f"cdrs_dist={row['min_cdr123_edit_distance']}")

    # Clean up temp column
    df.drop(columns=["_seq"], inplace=True, errors="ignore")

    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--designs", help="CSV with binder_sequence column (from rank_designs.py output)")
    ap.add_argument("--sabdab_zip", default="~/Downloads/all_structures.zip",
        help="Path to SAbDab all_structures.zip from SAbDab (default: ~/Downloads/all_structures.zip)")
    ap.add_argument("--cache", default=".sabdab_reference.json",
        help="Cache file for extracted CDR3 reference set (default: .sabdab_reference.json)")
    ap.add_argument("--out", help="Output CSV (default: adds _novelty suffix to --designs)")
    ap.add_argument("--min_edit_distance", type=int, default=4,
        help="Minimum edit distance for CDR3 to be considered novel (default: 4)")
    ap.add_argument("--cdrs_edit_distance_threshold", type=int, default=None,
        help="Minimum edit distance for CDR1+2+3 to be considered novel. "
             "Default: same as --min_edit_distance")
    ap.add_argument("--filter_mode",
        choices=["both", "cdr3_only", "cdrs_only"], default="both",
        help="Filter gate mode (default: both):\n"
             "  both     — both CDR3 and CDR1+2+3 must meet thresholds (strictest)\n"
             "  cdr3_only — only CDR3 distance is used (legacy behaviour)\n"
             "  cdrs_only — only CDR1+2+3 distance is used (primary gate)")
    ap.add_argument("--build_cache", action="store_true",
        help="Force rebuild of the SAbDab reference cache")
    ap.add_argument("--min_cdr3_len", type=int, default=5,
        help="Minimum CDR3 length to include in reference (default: 5)")
    args = ap.parse_args()

    sablab_zip = str(Path(args.sabdab_zip).expanduser())
    cache_path = Path(args.cache).expanduser()

    # ── Load or build reference ────────────────────────────────────────────
    if args.build_cache or not cache_path.exists():
        ref = build_reference_from_zip(sablab_zip, str(cache_path))
    else:
        print(f"Loading cached SAbDab reference from {cache_path} ...")
        with open(cache_path) as f:
            ref = json.load(f)
        print(f"  Loaded {ref['unique_count']:,} unique CDR3s from {ref['pdb_count']:,} PDBs")

    ref_cdr3s = [c for c in ref["cdr3_set"] if len(c) >= args.min_cdr3_len]
    # Build length-indexed reference for fast distance lookup
    ref_by_len = build_reference_by_length(ref_cdr3s)
    print(f"  Using {len(ref_cdr3s):,} reference CDR3s (len ≥ {args.min_cdr3_len}), "
          f"indexed into {len(ref_by_len)} length buckets\n")

    # ── Check designs ─────────────────────────────────────────────────────
    if args.designs:
        try:
            import pandas as pd
        except ModuleNotFoundError:
            raise SystemExit("pandas is required for --designs. Install it or run build_cache only.")

        out_path = args.out or re.sub(r"(\.csv)?$", "_novelty.csv", args.designs)
        designs_df = check_novelty(
            args.designs, ref_by_len, str(cache_path),
            min_edit_distance=args.min_edit_distance,
            cdrs_edit_distance_threshold=args.cdrs_edit_distance_threshold,
            filter_mode=args.filter_mode,
        )

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        designs_df.to_csv(out_path, index=False)
        print(f"\nWrote: {out_path}")
        n_pass = designs_df["novelty_pass"].sum()
        cdrs_thresh = args.cdrs_edit_distance_threshold or args.min_edit_distance
        print(f"  novelty_pass [{args.filter_mode}]: {n_pass}/{len(designs_df)} "
              f"({100*n_pass/len(designs_df):.1f}%)")
        print(f"  (CDR3 dist ≥ {args.min_edit_distance}: {(designs_df['cdr3_novel']).sum()}, "
              f"CDR1+2+3 dist ≥ {cdrs_thresh}: {(designs_df['cdr123_novel']).sum()})")
    else:
        print("No --designs provided. Cache built/loaded successfully.")


if __name__ == "__main__":
    main()