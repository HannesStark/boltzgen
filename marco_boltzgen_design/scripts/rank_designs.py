#!/usr/bin/env python3
from __future__ import annotations
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd

AA_CHARGE = {"K":1,"R":1,"H":0.1,"D":-1,"E":-1}
HYDROPHOBIC = set("AILMFWVY")


def infer_sequence(row):
    for k in ["designed_sequence", "binder_sequence", "sequence", "seq"]:
        if k in row and isinstance(row[k], str):
            return row[k]
    return ""


def infer_contacted_residues(row, human_col, mouse_col):
    """Try to extract contacted residue info from the metrics CSV directly.

    Some BoltzGen outputs include per-design interface residue lists.
    Fall back to empty strings if not present.
    """
    human_res = str(row.get(human_col, "") or "")
    mouse_res = str(row.get(mouse_col, "") or "")
    return human_res, mouse_res


def parse_contacts_from_metrics(df):
    """Detect whether the metrics CSV already contains contacted_residues columns.

    Returns the df with contacted_residues_human/mouse columns added (as comma-
    separated strings) if found, otherwise as empty columns.
    """
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
        print("  To enable it, either (a) pass --contacts CSV or (b) add contacted_residues_human/mouse columns to your metrics CSV.")
    return df, has_contacts


def has_nglyc(seq: str) -> bool:
    return bool(re.search(r"N[^P][ST]", seq))


def parse_set(s: str):
    if not s:
        return set()
    return {x.strip() for x in s.split(",") if x.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="BoltzGen all_designs_metrics.csv or aggregate_metrics_analyze.csv")
    ap.add_argument("--contacts", default="", help="Optional CSV with design_id,contacted_residues_human,contacted_residues_mouse")
    ap.add_argument("--human-conserved", default="", help="Comma-separated conserved human residues e.g. A:340,A:344")
    ap.add_argument("--mouse-conserved", default="", help="Comma-separated conserved mouse residues e.g. A:337,A:341")
    ap.add_argument("--max_len", type=int, default=120)
    ap.add_argument("--out", default="results/ranked_candidates.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.metrics)
    if "design_id" not in df.columns:
        df["design_id"] = np.arange(len(df)).astype(str)

    df["binder_sequence"] = df.apply(infer_sequence, axis=1)
    df["binder_length"] = df["binder_sequence"].str.len().fillna(0).astype(int)
    df["has_cys"] = df["binder_sequence"].str.contains("C", na=False)
    df["net_charge"] = df["binder_sequence"].apply(lambda s: sum(AA_CHARGE.get(a,0) for a in s) if isinstance(s,str) else 0)
    df["frac_hydrophobic"] = df["binder_sequence"].apply(lambda s: (sum(a in HYDROPHOBIC for a in s)/len(s)) if isinstance(s,str) and s else 0)
    df["nglyc_motif"] = df["binder_sequence"].apply(has_nglyc)
    df["too_long"] = df["binder_length"] > args.max_len
    df["excess_positive_charge"] = df["net_charge"] > 8
    df["hydrophobic_patch_flag"] = df["frac_hydrophobic"] > 0.5

    human_cons = parse_set(args.human_conserved)
    mouse_cons = parse_set(args.mouse_conserved)

    # Try to load external contacts file (optional).
    # If absent, try to detect contact columns directly in the metrics CSV.
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

    conf_cols = [c for c in ["ptm", "iptm", "plddt", "confidence", "ranking_score", "pLDDT"] if c in df.columns]
    base_conf = df[conf_cols].mean(axis=1) if conf_cols else 0
    penalties = (
      df["has_cys"].astype(int) + df["too_long"].astype(int) +
      df["excess_positive_charge"].astype(int) + df["hydrophobic_patch_flag"].astype(int) +
      df["nglyc_motif"].astype(int)
    )
    df["final_score"] = base_conf + 0.5 * df["crossreactivity_score"] - penalties

    ranked = df.sort_values("final_score", ascending=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(ranked)} rows")


if __name__ == "__main__":
    main()
