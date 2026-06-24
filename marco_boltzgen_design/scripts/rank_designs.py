#!/usr/bin/env python3
"""
rank_designs.py — Rank BoltzGen nanobody candidates by confidence, developability, and epitope coverage.

Extends the standard BoltzGen ranking with a BoltzProt-1-aligned developability panel:

  1. Basic developability filters
     • Length (max_len, default 120)
     • Cysteine count (free Cys → disulfide risk)
     • Net charge (|q| > 8 → solubility/self-association risk)
     • Hydrophobic fraction (> 0.42 → aggregation/HIC risk)
     • Aromatic fraction (> 0.14 → polyspecificity/BVP risk)
     • pI region (acidic/basic → HIC retention risk)
     • Hydrophobic patch (multiple adjacent hydrophobic runs → patchy aggregation)
     • Proline in CDR3 (structural disruption → Tm/thermal stability risk)
     • N-glycosylation sequons (N[^P][ST] → glycan heterogeneity)

  2. Cross-reactivity scoring
     • Human/Mouse conserved-residue contact count (via --human-conserved / --mouse-conserved)

  3. Confidence score
     • pLDDT / ipTM / ptm / ranking_score (BoltzGen confidence metrics)

Final score = base_confidence + α × crossreactivity_score + β × affinity_signal − penalties

where:
  α (--alpha_crossreactivity)  — cross-reactivity weight  (default: 0.5)
  β (--affinity_weight)        — interface quality weight (default: 0.3)
                                 (ipTM + interface H-bonds + salt bridges + buried SA)
  Penalties: cysteine, length, charge, hydrophobicity, N-glyc (2×), aromatic (2×),
             proline-in-CDR3 (2×)

Output columns include per-flag boolean columns and a human-readable `developability_flags`
summary column listing all issues for quick review.

Usage:
  python scripts/rank_designs.py --metrics results/all_designs_metrics.csv

  # With epitope coverage and tuned alpha (more cross-reactive bias):
  python scripts/rank_designs.py --metrics results/all_designs_metrics.csv \
    --human-conserved A:423,A:425,A:432,A:461,A:467,A:469,A:489,A:500 \
    --mouse-conserved A:6,A:8,A:15,A:44,A:50,A:52,A:72,A:83 \
    --alpha_crossreactivity 0.8

  # With interface-quality signal (default β=0.3, use 0 to disable):
  python scripts/rank_designs.py --metrics results/all_designs_metrics.csv \
    --affinity_weight 0.3

  # Load contacts from external CSV:
  python scripts/rank_designs.py --metrics results/all_designs_metrics.csv \
    --contacts results/interface_residues.csv
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ── Amino-acid property sets ─────────────────────────────────────────────────
AA_CHARGE = {"K": 1, "R": 1, "H": 0.1, "D": -1, "E": -1}
HYDROPHOBIC = set("AILMFWVY")
AROMATIC = set("FWY")          # correlates with polyspecificity (BVP ELISA proxy)
PROLINE = set("P")
NGLEC_PATTERN = re.compile(r"N[^P][ST]")
# 32-motif list matching Boltz API excluded_sequence_motifs exactly
NGLEC_MOTIFS = [
    "NAS", "NAT", "NCS", "NCT", "NDS", "NDT",
    "NES", "NET", "NFS", "NFT", "NGS", "NGT",
    "NHS", "NHT", "NIS", "NIT", "NKS", "NKT",
    "NLS", "NLT", "NMS", "NMT", "NNS", "NNT",
    "NQS", "NQT", "NRS", "NRT", "NSS", "SST",
    "NTS", "NTT", "NVS", "NVT", "NWS", "NWT",
    "NYS", "NYT",
]


# ── Helper functions ──────────────────────────────────────────────────────────

def infer_sequence(row):
    for k in ("designed_sequence", "binder_sequence", "sequence", "seq"):
        if k in row and isinstance(row[k], str):
            return row[k]
    return ""


def extract_cdr3(seq: str) -> str:
    """Extract CDR3 loop from a VHH sequence using FR4 motif as anchor."""
    if not seq:
        return ""
    fr4_match = re.search(r"WGQGT[LI]VTVSS", seq)
    if fr4_match:
        fr4_start = fr4_match.start()
        last_c = seq.rfind("C", 0, fr4_start)
        if last_c > 70:  # genuine CDR3 start position (not FR1 C)
            return seq[last_c + 1 : fr4_start]
    return ""


def has_cdr_cys(full_seq: str) -> bool:
    """Flag only Cysteines INSIDE CDR regions (not the conserved FR1 C at position ~22).

    VHH structure (approximate Kabat 0-indexed positions):
      FR1:    0-25   (ends at WG motif)
      CDR1:  ~26-38  (starts after first WG, ends before second WG)
      FR2:   ~39-54  (ends at WG motif)
      CDR2:  ~55-65  (starts after second WG, ~10 aa)
      FR3:   ~66-92
      CDR3:  ~93-110 (starts at last C before WGQGT, ends at WGQGT)
      FR4:   ~111-end

    Framework C positions:
      - FR1 conserved C at ~position 22 (anchor for CDR1)
      - CDR3 start C at ~position 93-97
    """
    if not full_seq:
        return False
    L = len(full_seq)

    # Extract CDR3
    cdr3 = extract_cdr3(full_seq)

    # Extract CDR1: after first WG, before second WG
    wg_positions = [m.start() for m in re.finditer(r"WG", full_seq[:70])]
    cdr1 = ""
    if len(wg_positions) >= 2:
        cdr1 = full_seq[wg_positions[0] + 2 : wg_positions[1]]

    # Extract CDR2: after second WG, ~12 aa
    cdr2 = ""
    if len(wg_positions) >= 2:
        cdr2 = full_seq[wg_positions[1] + 2 : wg_positions[1] + 14]

    # Flag if any CDR contains a C
    for cdr in (cdr1, cdr2, cdr3):
        if cdr and "C" in cdr:
            return True
    return False


def has_nglyc(seq: str) -> bool:
    return bool(NGLEC_PATTERN.search(seq or ""))


def frac_aromatic(seq: str) -> float:
    if not seq:
        return 0.0
    return sum(a in AROMATIC for a in seq) / len(seq)


def net_charge(seq: str) -> float:
    return sum(AA_CHARGE.get(a, 0) for a in (seq or ""))


def pi_region(seq: str) -> str:
    q = net_charge(seq)
    if q > 5:
        return "basic"
    elif q < -5:
        return "acidic"
    return "neutral"


def hydrophobic_patches(seq: str, run_len: int = 4) -> int:
    """Count hydrophobic runs of length >= run_len. Multiple patches → aggregation risk."""
    if not seq:
        return 0
    count = 0
    run = 0
    for aa in seq.upper():
        if aa in HYDROPHOBIC:
            run += 1
            if run >= run_len:
                count += 1
        else:
            run = 0
    return count


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


def parse_set(s: str):
    if not s:
        return set()
    return {x.strip() for x in s.split(",") if x.strip()}


def parse_contacts_from_metrics(df):
    human_col_candidates = ["contacted_residues_human", "contacted_human", "interface_residues_human"]
    mouse_col_candidates = ["contacted_residues_mouse", "contacted_mouse", "interface_residues_mouse"]

    h_col = next((c for c in human_col_candidates if c in df.columns), None)
    m_col = next((c for c in mouse_col_candidates if c in df.columns), None)

    df["contacted_residues_human"] = df[h_col].astype(str) if h_col else ""
    df["contacted_residues_mouse"] = df[m_col].astype(str) if m_col else ""

    has_contacts = bool(h_col and m_col)
    if has_contacts:
        print(f"  Detected contact columns in metrics: {h_col}, {m_col}")
    else:
        print("  No contact columns in metrics CSV — cross-reactivity scoring will be 0.")
        print("  To enable it, pass --contacts CSV or add contacted_residues_human/mouse columns.")
    return df, has_contacts


def build_flag_summary(row) -> str:
    """Human-readable list of all developability issues."""
    flags = []
    if row.get("has_cys"):
        flags.append("Cys")
    if row.get("nglyc_motif"):
        flags.append("N-glyc")
    if row.get("too_long"):
        flags.append("long")
    if row.get("excess_positive_charge"):
        flags.append("high_charge")
    if row.get("hydrophobic_patch_flag"):
        flags.append("hydro_patch")
    if row.get("aromatic_high"):
        flags.append("high_aromatic")
    if row.get("pi_acidic"):
        flags.append("acidic_pI")
    if row.get("pi_basic"):
        flags.append("basic_pI")
    if row.get("proline_cdr3"):
        flags.append("Pro_in_CDR3")
    return "; ".join(flags) if flags else "OK"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--metrics", required=True,
        help="BoltzGen all_designs_metrics.csv or aggregate_metrics_analyze.csv")
    ap.add_argument("--contacts", default="",
        help="Optional CSV: design_id,contacted_residues_human,contacted_residues_mouse")
    ap.add_argument("--human-conserved", default="",
        help="Comma-separated conserved human residues, e.g. A:423,A:425,...")
    ap.add_argument("--mouse-conserved", default="",
        help="Comma-separated conserved mouse residues, e.g. A:6,A:8,...")
    ap.add_argument("--max-len", type=int, default=120,
        help="Maximum binder length (default: 120)")
    ap.add_argument("--frac-hydro-max", type=float, default=0.42,
        help="Max hydrophobic fraction (default: 0.42 — BoltzProt-1 HIC proxy)")
    ap.add_argument("--frac-aro-max", type=float, default=0.14,
        help="Max aromatic fraction (default: 0.14 — BVP/polyspecificity proxy)")
    ap.add_argument("--alpha_crossreactivity", type=float, default=0.5,
        help="Weight for cross-reactivity score (default: 0.5). Tune up for more cross-reactive designs, down for human-selective.")
    ap.add_argument("--affinity_weight", type=float, default=0.3,
        help="Weight for interface-quality/affinity signal (default: 0.3). Set to 0 to disable. Uses ipTM, H-bonds, salt bridges, and buried SA from PLIP metrics.")
    ap.add_argument("--out", default="results/ranked_candidates.csv",
        help="Output CSV (default: results/ranked_candidates.csv)")
    # ── Quality pre-filter gates ─────────────────────────────────────────────
    ap.add_argument("--min_iptm", type=float, default=0.0,
        help="Hard gate: only rank designs with ipTM >= this value (default: 0, i.e. no filter). "
             "Recommended: 0.25-0.30 to remove non-binders before ranking.")
    ap.add_argument("--max_pae", type=float, default=0.0,
        help="Hard gate: only rank designs with min_interaction_pae <= this value (default: 0, no filter). "
             "Recommended: 12-15 Å to enforce close interface geometry.")
    ap.add_argument("--max_gly_ala_frac", type=float, default=1.0,
        help="Hard gate: only rank designs where CDR3 Gly+Ala fraction <= this value (default: 1.0, no filter). "
             "Recommended: 0.30-0.35 to exclude overly Gly/Ala-rich synthetic loops.")
    args = ap.parse_args()

    df = pd.read_csv(args.metrics)
    if "design_id" not in df.columns:
        df["design_id"] = np.arange(len(df)).astype(str)

    # ── Sequence inference ───────────────────────────────────────────────────
    df["binder_sequence"] = df.apply(infer_sequence, axis=1)

    # ── Basic sequence metrics ────────────────────────────────────────────────
    df["binder_length"] = df["binder_sequence"].str.len().fillna(0).astype(int)
    df["net_charge"] = df["binder_sequence"].apply(net_charge)
    df["frac_hydrophobic"] = df["binder_sequence"].apply(
        lambda s: sum(a in HYDROPHOBIC for a in s) / len(s) if s else 0
    )
    df["frac_aromatic"] = df["binder_sequence"].apply(frac_aromatic)

    # ── Developability flag columns ──────────────────────────────────────────
    # has_cys: only flag extra C's INSIDE CDR loops, not conserved framework C's
    # (FR1 C at pos ~22 is structural; CDR3-start C at pos ~93-97 is structural)
    if "designed_chain_sequence" in df.columns:
        df["has_cys"] = df["designed_chain_sequence"].apply(has_cdr_cys)
    else:
        df["has_cys"] = df["binder_sequence"].apply(has_cdr_cys)

    # N-glycosylation — 32-motif list (Boltz API match)
    df["nglyc_motif"] = df["binder_sequence"].apply(has_nglyc)

    # Length
    df["too_long"] = df["binder_length"] > args.max_len

    # Charge
    df["excess_positive_charge"] = df["net_charge"] > 8
    df["pi_acidic"] = df["binder_sequence"].apply(lambda s: pi_region(s) == "acidic")
    df["pi_basic"] = df["binder_sequence"].apply(lambda s: pi_region(s) == "basic")

    # Hydrophobicity (HIC proxy)
    df["hydrophobic_patch_flag"] = df["frac_hydrophobic"] > args.frac_hydro_max

    # Aromatic fraction (BVP/polyspecificity proxy; BoltzProt-1 Figure 5d)
    df["aromatic_high"] = df["frac_aromatic"] > args.frac_aro_max

    # Proline in CDR3 (thermal stability / Tm risk; Pro disrupts the β-sheet scaffold)
    df["proline_cdr3"] = df["binder_sequence"].apply(proline_in_cdr3)

    # ── Cross-reactivity scoring ──────────────────────────────────────────────
    human_cons = parse_set(args.human_conserved)
    mouse_cons = parse_set(args.mouse_conserved)

    if args.contacts:
        cdf = pd.read_csv(args.contacts)
        df = df.merge(cdf, on="design_id", how="left")
        print(f"  Loaded {len(cdf)} contact records from --contacts file.")
    else:
        df, has_contacts = parse_contacts_from_metrics(df)

    for col in ["contacted_residues_human", "contacted_residues_mouse"]:
        if col not in df.columns:
            df[col] = ""

    df["crossreactivity_score"] = df.apply(
        lambda r: (
            len(parse_set(str(r["contacted_residues_human"])) & human_cons)
            + len(parse_set(str(r["contacted_residues_mouse"])) & mouse_cons)
        ), axis=1
    )

    # ── Confidence score ──────────────────────────────────────────────────────
    conf_cols = [c for c in ["ptm", "iptm", "plddt", "confidence", "ranking_score", "pLDDT"]
                 if c in df.columns]
    base_conf = df[conf_cols].mean(axis=1) if conf_cols else 0

    # ── Interface quality / affinity signal ──────────────────────────────────
    # Combines ipTM (binding specificity), PLIP H-bonds, salt bridges, and
    # buried surface area — all from BoltzGen folding analysis step.
    # Higher = better interface. Normalised to [0, 1] per column before summing.
    def _norm(s, col, lo, hi):
        """Min-max normalise a column, clamping outliers to [0, 1]."""
        if col not in s.columns:
            return 0.0
        v = s[col].fillna(0)
        return ((v - lo) / (hi - lo)).clip(0, 1)

    iq = pd.DataFrame(index=df.index)
    iq["iptm"] = _norm(df, "iptm", 0.5, 0.9)          # ipTM: 0.5→0.9 maps to 0→1
    iq["hbonds"] = _norm(df, "plip_hbonds_refolded", 0, 10)   # up to 10 H-bonds
    iq["sb"]    = _norm(df, "plip_saltbridges_refolded", 0, 5) # up to 5 salt bridges
    iq["bsa"]   = _norm(df, "delta_sasa_refolded", 200, 1200)  # buried SA 200→1200 Å²
    # Mean of available components (handles missing columns gracefully)
    iface_cols = [c for c in iq.columns if c in df.columns]
    df["interface_quality"] = iq[iface_cols].mean(axis=1) if iface_cols else 0.0
    n_iface = (iq[iface_cols] > 0).sum(axis=1) if iface_cols else 0
    df["interface_quality"] = df["interface_quality"].where(n_iface > 0, 0.0)

    # ── CDR3 Gly/Ala fraction (synthetic-loop detector) ──────────────────────
    def gly_ala_frac(row) -> float:
        cdr3 = extract_cdr3(str(row.get("designed_chain_sequence", "")))
        if len(cdr3) == 0:
            cdr3 = str(row.get("binder_sequence", ""))
        if not cdr3:
            return 0.0
        return (cdr3.count("G") + cdr3.count("A")) / len(cdr3)

    df["cdr3_gly_ala_frac"] = df.apply(gly_ala_frac, axis=1)

    # ── Quality pre-filter gates ─────────────────────────────────────────────
    # Applied BEFORE ranking to remove non-binders and poor-quality designs.
    n_before = len(df)
    gate_iptm = df["iptm"] >= args.min_iptm if args.min_iptm > 0 else pd.Series(True, index=df.index)
    gate_pae  = df["min_interaction_pae"] <= args.max_pae if args.max_pae > 0 else pd.Series(True, index=df.index)
    gate_ga   = df["cdr3_gly_ala_frac"] <= args.max_gly_ala_frac

    df["_gate_pass"] = gate_iptm & gate_pae & gate_ga
    n_after = int(df["_gate_pass"].sum())

    if args.min_iptm > 0 or args.max_pae > 0 or args.max_gly_ala_frac < 1.0:
        print(f"\n=== Quality pre-filter gates ===")
        if args.min_iptm > 0:
            print(f"  ipTM < {args.min_iptm}:  {(~gate_iptm).sum()} removed")
        if args.max_pae > 0:
            print(f"  PAE > {args.max_pae} A:  {(~gate_pae).sum()} removed")
        if args.max_gly_ala_frac < 1.0:
            print(f"  Gly+Ala frac > {args.max_gly_ala_frac}: {(~gate_ga).sum()} removed")
        print(f"  Designs after gating: {n_after}/{n_before}")

    # Remove failed designs (keep those that pass all gates)
    df = df[df["_gate_pass"]].copy()
    # Each flag = -1 penalty. N-glyc and Pro_CDR3 penalized more (known developability issues)
    penalties = (
        df["has_cys"].astype(int)
        + df["too_long"].astype(int)
        + df["excess_positive_charge"].astype(int)
        + df["hydrophobic_patch_flag"].astype(int)
        + df["nglyc_motif"].astype(int) * 2          # strong penalty — glycan heterogeneity
        + df["aromatic_high"].astype(int)
        + df["pi_acidic"].astype(int)
        + df["pi_basic"].astype(int)
        + df["proline_cdr3"].astype(int) * 2         # strong penalty — Tm disruption
    )
    df["final_score"] = (
        base_conf
        + args.alpha_crossreactivity * df["crossreactivity_score"]
        + args.affinity_weight * df["interface_quality"]
        - penalties
    )
    df["base_confidence"] = base_conf
    df["developability_penalties"] = penalties

    # Human-readable flag summary
    df["developability_flags"] = df.apply(build_flag_summary, axis=1)

    # ── Sort and save ─────────────────────────────────────────────────────────
    ranked = df.sort_values("final_score", ascending=False).reset_index(drop=True)

    out_cols = [
        "design_id", "binder_sequence", "binder_length",
        "final_score", "base_confidence", "crossreactivity_score",
        "interface_quality",
        "developability_penalties", "developability_flags",
        # flag detail
        "has_cys", "nglyc_motif", "too_long",
        "excess_positive_charge", "hydrophobic_patch_flag",
        "aromatic_high", "pi_acidic", "pi_basic", "proline_cdr3",
        "frac_hydrophobic", "frac_aromatic", "net_charge",
        # contact info
        "contacted_residues_human", "contacted_residues_mouse",
        # original metrics
    ]
    # Append any original columns not yet included (preserve BoltzGen output)
    for col in df.columns:
        if col not in out_cols and col not in ("_seq",):
            out_cols.append(col)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    ranked[[c for c in out_cols if c in ranked.columns]].to_csv(args.out, index=False)

    n_pass = (~ranked["nglyc_motif"]).sum()
    print(f"\nWrote {args.out} ({len(ranked)} candidates)")
    print(f"  N-glyc removed : {ranked['nglyc_motif'].sum()} ({100*ranked['nglyc_motif'].mean():.1f}%)")
    print(f"  Hydrophobic    : {ranked['hydrophobic_patch_flag'].sum()}")
    print(f"  Aromatic high  : {ranked['aromatic_high'].sum()}")
    print(f"  pI acidic      : {ranked['pi_acidic'].sum()}")
    print(f"  pI basic       : {ranked['pi_basic'].sum()}")
    print(f"  Pro in CDR3    : {ranked['proline_cdr3'].sum()}")
    print(f"  Pass all       : {n_pass} ({100*n_pass/len(ranked):.1f}%)")
    iq_vals = ranked["interface_quality"]
    ga_vals = ranked["cdr3_gly_ala_frac"]
    print(f"  Interface quality: mean={iq_vals.mean():.3f}  median={iq_vals.median():.3f}  max={iq_vals.max():.3f}")
    print(f"  CDR3 Gly+Ala frac: mean={ga_vals.mean():.3f}  max={ga_vals.max():.3f}")
    print(f"  Score params:  alpha (crossreactivity) = {args.alpha_crossreactivity}  beta (affinity) = {args.affinity_weight}")
    if args.min_iptm > 0 or args.max_pae > 0 or args.max_gly_ala_frac < 1.0:
        print(f"  Quality gates: min_iptm={args.min_iptm}  max_pae={args.max_pae}  max_gly_ala_frac={args.max_gly_ala_frac}")

    print(f"\n=== Quality summary (top 10 by final_score) ===")
    top10 = ranked.head(10)[["design_id","final_score","iptm","min_interaction_pae",
                              "cdr3_gly_ala_frac","developability_flags"]]
    print(top10.to_string(index=False))


if __name__ == "__main__":
    main()