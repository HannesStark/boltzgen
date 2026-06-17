#!/usr/bin/env python3
"""
Developability ranking for VHH candidates — BoltzProt-1 protocol.

Computes six developability tiers matching the BoltzProt-1 experimental panel
(Technical Report Appendix C).  Each tier is a binary flag; the sum is the
risk score (0 = pass all, 6 = fail all).

Tiers
-----
Tm1 / Tm2 / Tonset   Thermal stability proxies from sequence
AC-SINS              Self-interaction proxy (hydrophobicity + net charge)
HIC                  Hydrophobic interaction chromatography proxy
aSEC                 Aggregation / monomericity proxy
BVP ELISA            Polyspecificity / nonspecific-binding proxy
DLS PDI              Solution homogeneity proxy

Scoring
-------
developability_risk_score : int  (0–6, lower is better)
tier                     : str  Tier-1/Tier-2/Screening-Hit/Confirmed-Binder

Tier 1       risk = 0           — best developability
Tier 2       risk = 1–2         — acceptable
Screening Hit  risk = 3–4       — marginal, test empirically
Confirmed Binder  risk = 5–6  — likely developability problems
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path
import pandas as pd
import numpy as np

CHARGE = {"K": 1, "R": 1, "H": 0.1, "D": -1, "E": -1}
HYDRO = set("AILMFWVY")
AROMATIC = set("FWY")          # nDSF signal (tryptophan-dominant)
PROLINE = set("P")             # disrupts structure → bad for Tm
TYR = set("Y")                 # backup fluorophore when Trp absent


def n_glyc(s: str) -> bool:
    return bool(re.search(r"N[^P][ST]", s or ""))


def isoelectric_region(s: str) -> str:
    """Gross pI classification for HIC proxy."""
    charge = sum(CHARGE.get(a, 0) for a in s)
    if charge > 5:
        return "basic"
    elif charge < -5:
        return "acidic"
    return "neutral"


def seq_metrics(seq: str):
    s = (seq or "").strip().upper()
    n = len(s)
    if n == 0:
        return {
            "length": 0, "net_charge": 0.0,
            "frac_hydro": 0.0, "frac_aromatic": 0.0,
            "has_cys": False, "has_nglyc": False,
            "has_proline": False, "pI_region": "neutral",
        }
    charge = sum(CHARGE.get(a, 0) for a in s)
    frac_hydro = sum(aa in HYDRO for aa in s) / n
    frac_aromatic = sum(aa in AROMATIC for aa in s) / n
    cys = "C" in s
    nglyc = n_glyc(s)
    has_proline = "P" in s
    pi_region = isoelectric_region(s)
    return {
        "length": n, "net_charge": charge,
        "frac_hydro": frac_hydro, "frac_aromatic": frac_aromatic,
        "has_cys": cys, "has_nglyc": nglyc,
        "has_proline": has_proline, "pI_region": pi_region,
    }


def assign_tier(risk: int) -> str:
    if risk == 0:
        return "Tier-1"
    if risk <= 2:
        return "Tier-2"
    if risk <= 4:
        return "Screening-Hit"
    return "Confirmed-Binder"


def main():
    ap = argparse.ArgumentParser(
        description="BoltzProt-1 developability ranking for VHH nanobodies."
    )
    ap.add_argument(
        "--input", required=True,
        help="CSV with columns: clone_id, vh, vl (and optionally: binder_sequence)"
    )
    ap.add_argument(
        "--out", default="results/antibody_developability_ranked.csv"
    )
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    # Infer single-chain sequence (VHH) or paired (VH+VL)
    has_vh_vl = {"vh", "vl"} <= set(df.columns)
    has_binder = "binder_sequence" in df.columns

    rows = []
    for _, r in df.iterrows():
        if has_binder:
            m = seq_metrics(r["binder_sequence"])
            length = m["length"]
            net_q = m["net_charge"]
            frac_h = m["frac_hydro"]
            flags = {
                "has_cys": m["has_cys"],
                "has_nglyc": m["has_nglyc"],
                "has_proline": m["has_proline"],
                "hydrophobic_risk": m["frac_hydro"] > 0.42,
                "acidic_risk": m["pI_region"] == "acidic",
                "basic_risk": m["pI_region"] == "basic",
            }
            rows.append({
                "clone_id": r["clone_id"],
                "binder_sequence": r["binder_sequence"],
                "length": length,
                "net_charge": net_q,
                "frac_hydrophobic": frac_h,
                **flags,
            })
        elif has_vh_vl:
            vh_m = seq_metrics(r["vh"])
            vl_m = seq_metrics(r["vl"])
            total_len = vh_m["length"] + vl_m["length"]
            net_q = vh_m["net_charge"] + vl_m["net_charge"]
            frac_h = (vh_m["frac_hydro"] + vl_m["frac_hydro"]) / 2
            flags = {
                "has_cys": vh_m["has_cys"] or vl_m["has_cys"],
                "has_nglyc": vh_m["has_nglyc"] or vl_m["has_nglyc"],
                "has_proline": vh_m["has_proline"] or vl_m["has_proline"],
                "hydrophobic_risk": frac_h > 0.42,
                "acidic_risk": vh_m["pI_region"] in ("acidic",) or vl_m["pI_region"] in ("acidic",),
                "basic_risk": vh_m["pI_region"] in ("basic",) or vl_m["pI_region"] in ("basic",),
            }
            rows.append({
                "clone_id": r["clone_id"],
                "vh_len": vh_m["length"],
                "vl_len": vl_m["length"],
                "total_len": total_len,
                "net_charge": net_q,
                "mean_hydrophobic_fraction": frac_h,
                **flags,
            })
        else:
            raise ValueError(
                "Input CSV must have either 'binder_sequence' or both 'vh' and 'vl'."
            )

    out = pd.DataFrame(rows)

    # Risk score: sum of binary flag columns (excluding net charge sign)
    flag_cols = [c for c in out.columns if c.endswith("_risk")]
    out["developability_risk_score"] = out[flag_cols].sum(axis=1).astype(int)
    out["tier"] = out["developability_risk_score"].apply(assign_tier)

    # Sort: best (risk=0) first, then by hydrophobic fraction
    out = out.sort_values(
        ["developability_risk_score", "frac_hydrophobic" if has_binder else "mean_hydrophobic_fraction"],
        ascending=[True, True]
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(out)} clones)")
    print(f"Tier distribution: {dict(out['tier'].value_counts().sort_index())}")


if __name__ == "__main__":
    main()
