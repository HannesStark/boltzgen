#!/usr/bin/env python3
"""Find putative hotspot residues on human MARCO / mouse Marco SRCR domains.

Two modes:
1) Complex mode (recommended): provide predicted/experimental antibody-target complexes
   and extract recurring interface residues on target chains.
2) Apo mode: provide target-only structures and rank exposed + human/mouse-conserved
   residues as docking hotspot candidates.

The script is intentionally lightweight and depends only on gemmi from the BoltzGen env.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import gemmi

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


def read_structure(path: str) -> gemmi.Structure:
    p = Path(path)
    if p.suffix.lower() in {".cif", ".mmcif"}:
        return gemmi.read_structure(str(p))
    return gemmi.read_pdb(str(p))


def iter_poly_residues(chain: gemmi.Chain) -> Iterable[gemmi.Residue]:
    for res in chain:
        if res.entity_type == gemmi.EntityType.Polymer and res.name in AA3_TO_1:
            yield res


def residue_key(res: gemmi.Residue) -> Tuple[str, int, str]:
    rid = res.label_seq
    if rid is None:
        rid = res.seqid.num
    return (res.name, int(rid), res.seqid.icode.strip())


def get_chain(model: gemmi.Model, cid: str) -> gemmi.Chain:
    try:
        return model[cid]
    except Exception as exc:
        raise ValueError(f"Chain '{cid}' not found") from exc


def ca_position(res: gemmi.Residue):
    for a in res:
        if a.name == "CA":
            return a.pos
    return None


def extract_sequence(chain: gemmi.Chain) -> Tuple[str, List[Tuple[str, int, str]]]:
    seq = []
    keys = []
    for res in iter_poly_residues(chain):
        aa = AA3_TO_1.get(res.name)
        if aa:
            seq.append(aa)
            keys.append(residue_key(res))
    return "".join(seq), keys


def needleman_wunsch(a: str, b: str, match=2, mismatch=-1, gap=-2):
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    bt = [[None]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        dp[i][0] = i*gap
        bt[i][0] = 'U'
    for j in range(1, m+1):
        dp[0][j] = j*gap
        bt[0][j] = 'L'
    for i in range(1, n+1):
        for j in range(1, m+1):
            sdiag = dp[i-1][j-1] + (match if a[i-1] == b[j-1] else mismatch)
            sup = dp[i-1][j] + gap
            sleft = dp[i][j-1] + gap
            best = max(sdiag, sup, sleft)
            dp[i][j] = best
            bt[i][j] = 'D' if best == sdiag else ('U' if best == sup else 'L')
    i, j = n, m
    pairs = []
    while i > 0 or j > 0:
        move = bt[i][j]
        if move == 'D':
            pairs.append((i-1, j-1))
            i -= 1
            j -= 1
        elif move == 'U':
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def min_heavy_atom_distance(res_a: gemmi.Residue, res_b: gemmi.Residue) -> float:
    dmin = float('inf')
    for aa in res_a:
        if aa.element.name == 'H':
            continue
        for bb in res_b:
            if bb.element.name == 'H':
                continue
            d = aa.pos.dist(bb.pos)
            if d < dmin:
                dmin = d
    return dmin


def count_contacts(struct: gemmi.Structure, target_chain_id: str, binder_chain_ids: Sequence[str], cutoff: float) -> Dict[Tuple[str, int, str], int]:
    model = struct[0]
    tchain = get_chain(model, target_chain_id)
    bchains = [get_chain(model, x) for x in binder_chain_ids]
    counts = defaultdict(int)
    for tr in iter_poly_residues(tchain):
        key = residue_key(tr)
        for bc in bchains:
            contacted = False
            for br in iter_poly_residues(bc):
                if min_heavy_atom_distance(tr, br) <= cutoff:
                    contacted = True
                    break
            if contacted:
                counts[key] += 1
    return counts


def exposure_proxy(chain: gemmi.Chain, radius=10.0) -> Dict[Tuple[str, int, str], float]:
    residues = list(iter_poly_residues(chain))
    ca = [ca_position(r) for r in residues]
    scores = {}
    for i, ri in enumerate(residues):
        if ca[i] is None:
            continue
        neighbors = 0
        for j, rj in enumerate(residues):
            if i == j or ca[j] is None:
                continue
            if ca[i].dist(ca[j]) < radius:
                neighbors += 1
        # higher score = more exposed
        scores[residue_key(ri)] = 1.0 / (1.0 + neighbors)
    return scores


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--human-structure", required=True)
    ap.add_argument("--mouse-structure", required=True)
    ap.add_argument("--human-chain", default="A")
    ap.add_argument("--mouse-chain", default="A")
    ap.add_argument("--human-complexes", nargs="*", default=[])
    ap.add_argument("--mouse-complexes", nargs="*", default=[])
    ap.add_argument("--human-binder-chains", default="H,L", help="comma-separated")
    ap.add_argument("--mouse-binder-chains", default="H,L", help="comma-separated")
    ap.add_argument("--contact-cutoff", type=float, default=4.5)
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    h = read_structure(args.human_structure)
    m = read_structure(args.mouse_structure)
    hchain = get_chain(h[0], args.human_chain)
    mchain = get_chain(m[0], args.mouse_chain)

    hseq, hkeys = extract_sequence(hchain)
    mseq, mkeys = extract_sequence(mchain)
    align_pairs = needleman_wunsch(hseq, mseq)

    conserved_h = set()
    conserved_m = set()
    for ih, im in align_pairs:
        if hseq[ih] == mseq[im]:
            conserved_h.add(hkeys[ih])
            conserved_m.add(mkeys[im])

    hexp = exposure_proxy(hchain)
    mexp = exposure_proxy(mchain)

    hcont_total = defaultdict(int)
    mcont_total = defaultdict(int)
    hcomplex_count = len(args.human_complexes)
    mcomplex_count = len(args.mouse_complexes)

    for p in args.human_complexes:
        c = count_contacts(read_structure(p), args.human_chain, args.human_binder_chains.split(','), args.contact_cutoff)
        for k, v in c.items():
            hcont_total[k] += v

    for p in args.mouse_complexes:
        c = count_contacts(read_structure(p), args.mouse_chain, args.mouse_binder_chains.split(','), args.contact_cutoff)
        for k, v in c.items():
            mcont_total[k] += v

    rows = []
    for k in hkeys:
        aa, pos, ins = k
        cont_freq = (hcont_total[k] / max(hcomplex_count, 1)) if hcomplex_count else 0.0
        score = (0.55 * cont_freq) + (0.30 * hexp.get(k, 0.0)) + (0.15 * (1.0 if k in conserved_h else 0.0))
        rows.append(("human", f"{args.human_chain}:{pos}{ins}", AA3_TO_1[aa], cont_freq, hexp.get(k, 0.0), k in conserved_h, score))

    for k in mkeys:
        aa, pos, ins = k
        cont_freq = (mcont_total[k] / max(mcomplex_count, 1)) if mcomplex_count else 0.0
        score = (0.55 * cont_freq) + (0.30 * mexp.get(k, 0.0)) + (0.15 * (1.0 if k in conserved_m else 0.0))
        rows.append(("mouse", f"{args.mouse_chain}:{pos}{ins}", AA3_TO_1[aa], cont_freq, mexp.get(k, 0.0), k in conserved_m, score))

    rows.sort(key=lambda x: x[-1], reverse=True)
    rows = rows[: args.top_n * 2]

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["species", "residue", "aa", "contact_freq", "exposure_proxy", "conserved_human_mouse", "hotspot_score"])
        for r in rows:
            w.writerow(r)

    print(f"Wrote hotspot table: {outp}")


if __name__ == "__main__":
    main()
