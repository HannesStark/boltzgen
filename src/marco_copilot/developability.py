from __future__ import annotations

AA_CHARGE = {"D": -1, "E": -1, "K": 1, "R": 1}
HYDRO = set("AILMFWVY")


def features(seq: str) -> dict:
    s = seq.upper()
    length = len(s)
    net_charge = sum(AA_CHARGE.get(a, 0) for a in s)
    cys_count = s.count("C")
    nglyc = sum(1 for i in range(length - 2) if s[i] == "N" and s[i + 1] != "P" and s[i + 2] in {"S", "T"})
    hydrophobic_fraction = (sum(1 for a in s if a in HYDRO) / length) if length else 0.0
    # very rough pI estimate for MVP
    pI_est = 7.0 + (net_charge / max(length, 1)) * 10
    return {
        "length": length,
        "net_charge": net_charge,
        "pI_est": round(pI_est, 2),
        "cys_count": cys_count,
        "nglyc_motifs": nglyc,
        "hydrophobic_fraction": round(hydrophobic_fraction, 3),
    }
