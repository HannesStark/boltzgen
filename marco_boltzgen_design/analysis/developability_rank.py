#!/usr/bin/env python3
from __future__ import annotations
import argparse
import re
import pandas as pd

CHARGE = {"K": 1, "R": 1, "H": 0.1, "D": -1, "E": -1}
HYDRO = set("AILMFWVY")


def n_glyc(s: str) -> bool:
    return bool(re.search(r"N[^P][ST]", s or ""))


def seq_metrics(seq: str):
    s = (seq or "").strip().upper()
    n = len(s)
    charge = sum(CHARGE.get(a, 0) for a in s)
    frac_hydro = (sum(aa in HYDRO for aa in s) / n) if n else 0.0
    cys = "C" in s
    return n, charge, frac_hydro, cys, n_glyc(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV with columns: clone_id,vh,vl")
    ap.add_argument("--out", default="results/antibody_developability_ranked.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    required = {"clone_id", "vh", "vl"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows = []
    for _, r in df.iterrows():
        vh_n, vh_q, vh_h, vh_c, vh_nxst = seq_metrics(r["vh"])
        vl_n, vl_q, vl_h, vl_c, vl_nxst = seq_metrics(r["vl"])
        total_len = vh_n + vl_n
        net_q = vh_q + vl_q
        hydro = (vh_h + vl_h) / 2
        flags = {
            "has_cys": vh_c or vl_c,
            "has_nglyc": vh_nxst or vl_nxst,
            "too_long": total_len > 280,
            "too_charged": abs(net_q) > 20,
            "hydrophobic_risk": hydro > 0.42,
        }
        risk = sum(int(v) for v in flags.values())
        rows.append({
            "clone_id": r["clone_id"],
            "vh_len": vh_n,
            "vl_len": vl_n,
            "total_len": total_len,
            "net_charge": net_q,
            "mean_hydrophobic_fraction": hydro,
            **flags,
            "developability_risk_score": risk,
        })

    out = pd.DataFrame(rows).sort_values(["developability_risk_score", "mean_hydrophobic_fraction"])
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(out)} clones)")


if __name__ == "__main__":
    main()
