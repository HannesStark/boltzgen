#!/usr/bin/env python3
"""Validate BoltzGen designs by back-folding binder sequences with AlphaFold2/3
and comparing the result to the BoltzGen-predicted complex.

Usage:
    # Quick local (requires AF2 via colabfold or local installation):
    python scripts/validate_designs.py \
        --complexes runs/production_final/ \
        --out results/af_validation.csv \
        --method colabfold

    # HPC / full AF2 server:
    python scripts/validate_designs.py \
        --complexes runs/production_final/ \
        --out results/af_validation.csv \
        --method af2 \
        --af2_server https://alphafold.ebi.ac.uk \
        --api-key YOUR_ALPHAFOLD_API_KEY

Output CSV: design_id, af2_rmsd, af2_pae, af2_plddt, flag_ok
  flag_ok = True if af2_rmsd < 2.5 Å AND af2_pae < 5.0
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

import numpy as np
import pandas as pd

# Optional dependency check — AF2 backfolding is advisory, not fatal
try:
    import gemmi

    HAS_GEMMI = True
except ImportError:
    HAS_GEMMI = False
    print("WARNING: gemmi not available; CA RMSD computation will be skipped.", file=sys.stderr)


# ----------------------------------------------------------------------
# AlphaFold2 / ColabFold backfold + RMSD computation
# ----------------------------------------------------------------------


class AF2Result(NamedTuple):
    design_id: str
    af2_rmsd: float        # CA RMSD (Å) between AF2-backfold and BoltzGen design
    af2_pae: float         # mean Predicted Aligned Error (Å)
    af2_plddt: float       # mean pLDDT of binder
    flag_ok: bool
    error: str             # empty if OK


def run_colabfold_local(seq: str, work_dir: Path, timeout_s: int = 300) -> Optional[dict]:
    """Run ColabFold locally (requires `colabfold_batch` on $PATH).

    Returns a dict with keys 'pae', 'plddt', 'model_cif', or None on failure.
    """
    fasta_path = work_dir / "input.fasta"
    with open(fasta_path, "w") as f:
        f.write(f">query\n{seq}\n")

    result = subprocess.run(
        ["colabfold_batch", "--quiet", "--no-cache", str(work_dir), str(work_dir)],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        cwd=str(work_dir),
    )
    if result.returncode != 0:
        return None

    # ColabFold outputs to <input>.afa.db3 / *_score.json
    # Look for the first .cif / pdb result
    cif_files = sorted(work_dir.glob("*.cif"))
    json_files = sorted(work_dir.glob("*_score.json"))
    if not cif_files or not json_files:
        return None

    import json

    scores = json.load(json_files[0])
    return {
        "model_cif": cif_files[0],
        "pae": scores.get("pae", None),
        "plddt": scores.get("plddt", None),
    }


def run_af2_server(seq: str, server_url: str, api_key: str, work_dir: Path) -> Optional[dict]:
    """Submit a single sequence to an AlphaFold server and fetch the result.

    Args:
        seq:        binder amino acid sequence
        server_url: e.g. "https://alphafold.ebi.ac.uk"
        api_key:    optional API key for AF2 server
        work_dir:   scratch directory for downloading result
    """
    import json
    import time
    import urllib.request
    import urllib.parse

    # Submit job
    payload = json.dumps({"sequence": seq, "modelType": "auto"}).encode()
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        f"{server_url}/api/v2/predict",
        data=payload,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            job = json.loads(resp.read())
    except Exception as exc:
        return {"error": str(exc)}

    job_id = job.get("jobId") or job.get("job_id")
    if not job_id:
        return {"error": "No jobId in AF2 server response"}

    # Poll until done
    for _ in range(120):   # up to 10 min
        time.sleep(5)
        try:
            req2 = urllib.request.Request(
                f"{server_url}/api/v2/results/{job_id}",
                headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(req2, timeout=30) as resp:
                result = json.loads(resp.read())
                if result.get("status") == "completed":
                    # Download CIF
                    for entry in result.get("cif", []):
                        url = entry.get("url")
                        if url:
                            dest = work_dir / "af2_model.cif"
                            urllib.request.urlretrieve(url, dest)
                            return {"model_cif": dest, "pae": result.get("pae"), "plddt": result.get("plddt")}
                    return {"error": "No cif URL in AF2 result"}
                elif result.get("status") in ("failed", "error"):
                    return {"error": f"AF2 job {result.get('status')}"}
        except Exception as exc:
            return {"error": str(exc)}
    return {"error": "AF2 server polling timed out"}


def compute_ca_rmsd(cif_a: Path, cif_b: Path) -> float:
    """Compute CA-atom RMSD between two mmCIF structures using gemmi.

    Uses Cα pairs matched by residue sequence number.
    Returns RMSD in Å, or -1.0 if computation fails.
    """
    if not HAS_GEMMI:
        return -1.0

    try:
        import gemmi

        def get_cas(path: Path) -> list:
            st = gemmi.read_structure(str(path))
            ca_list = []
            for chain in st[0]:
                for res in chain:
                    for atom in res:
                        if atom.name == " CA ":
                            ca_list.append(atom.pos)
            return ca_list

        ca_a = get_cas(cif_a)
        ca_b = get_cas(cif_b)
        n = min(len(ca_a), len(ca_b))
        if n == 0:
            return -1.0

        rmsd = np.sqrt(
            sum((ca_a[i] - ca_b[i]).norm_sq() for i in range(n)) / n
        )
        return round(float(rmsd), 3)
    except Exception:
        return -1.0


def validate_design(
    design_id: str,
    binder_seq: str,
    boltzgen_complex_cif: Path,
    method: str,
    work_dir: Path,
    af2_server: str,
    api_key: str,
    rmsd_threshold: float = 2.5,
    pae_threshold: float = 5.0,
) -> AF2Result:
    """Run AF2 backfold on binder_seq and compare to BoltzGen design."""
    seq_clean = binder_seq.strip().upper()
    if not seq_clean:
        return AF2Result(design_id, -1.0, -1.0, -1.0, False, "empty binder sequence")

    tmp = work_dir / design_id
    tmp.mkdir(parents=True, exist_ok=True)

    try:
        if method == "colabfold":
            af2_out = run_colabfold_local(seq_clean, tmp)
            if af2_out is None:
                return AF2Result(design_id, -1.0, -1.0, -1.0, False, "colabfold failed")
        elif method == "af2":
            af2_out = run_af2_server(seq_clean, af2_server, api_key, tmp)
            if "error" in af2_out:
                return AF2Result(design_id, -1.0, -1.0, -1.0, False, af2_out["error"])
        else:
            return AF2Result(design_id, -1.0, -1.0, -1.0, False, f"unknown method: {method}")

        # Compute RMSD against BoltzGen design
        rmsd = compute_ca_rmsd(af2_out["model_cif"], boltzgen_complex_cif)
        pae = float(af2_out.get("pae", -1.0) or -1.0)
        plddt = float(af2_out.get("plddt", -1.0) or -1.0)

        flag_ok = (0 < rmsd < rmsd_threshold) and (0 < pae < pae_threshold)
        return AF2Result(design_id, rmsd, pae, plddt, flag_ok, "")

    except Exception as exc:
        return AF2Result(design_id, -1.0, -1.0, -1.0, False, str(exc))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def find_complexes(complex_dir: Path) -> Dict[str, Path]:
    """Return design_id → complex CIF path for designs in complex_dir."""
    complexes = {}
    for cif in complex_dir.glob("*.cif"):
        complexes[cif.stem] = cif
    return complexes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--complexes",
        type=Path,
        required=True,
        help="Directory containing BoltzGen-predicted complex CIF files",
    )
    ap.add_argument(
        "--metrics",
        type=Path,
        help="Optional: metrics CSV to get binder_sequence column (otherwise sequence inferred from CIF)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("results/af_validation.csv"),
        help="Output CSV (default: results/af_validation.csv)",
    )
    ap.add_argument(
        "--method",
        choices=["colabfold", "af2"],
        default="colabfold",
        help="AF2 backfold method (default: colabfold)",
    )
    ap.add_argument(
        "--af2_server",
        default="https://alphafold.ebi.ac.uk",
        help="AF2 server URL (used when --method=af2)",
    )
    ap.add_argument(
        "--api_key",
        default="",
        help="Optional API key for AF2 server",
    )
    ap.add_argument(
        "--rmsd_threshold",
        type=float,
        default=2.5,
        help="Max CA RMSD (Å) to flag a design as OK (default: 2.5)",
    )
    ap.add_argument(
        "--pae_threshold",
        type=float,
        default=5.0,
        help="Max mean PAE (Å) to flag a design as OK (default: 5.0)",
    )
    ap.add_argument(
        "--top_n",
        type=int,
        default=50,
        help="Validate only top-N designs by score (default: 50, use 0 for all)",
    )
    ap.add_argument(
        "--scratch",
        type=Path,
        default=Path("/tmp/af_validation"),
        help="Scratch directory for AF2 inputs/outputs (default: /tmp/af_validation)",
    )
    args = ap.parse_args()

    if not HAS_GEMMI:
        print("WARNING: gemmi not installed. CA RMSD will be -1.0. Install with: pip install gemmi", file=sys.stderr)

    # Load metrics if provided for sequence data
    seq_map: Dict[str, str] = {}
    if args.metrics and args.metrics.exists():
        df = pd.read_csv(args.metrics)
        if args.top_n > 0 and "final_score" in df.columns:
            df = df.sort_values("final_score", ascending=False).head(args.top_n)
        elif args.top_n > 0:
            df = df.head(args.top_n)

        for _, row in df.iterrows():
            did = str(row.get("design_id", ""))
            seq = str(row.get("designed_sequence", "")) or str(row.get("binder_sequence", ""))
            if did and seq:
                seq_map[did] = seq
        print(f"Loaded {len(seq_map)} sequences from metrics file.")
    else:
        print("WARNING: No metrics CSV — will not extract binder sequences. AF validation requires sequences.")

    complexes = find_complexes(args.complexes)
    print(f"Found {len(complexes)} complex CIF files in {args.complexes}")

    args.scratch.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows: List[AF2Result] = []
    for design_id, cif_path in complexes.items():
        if args.top_n > 0 and design_id not in seq_map:
            # Skip if not in top-N
            continue
        seq = seq_map.get(design_id, "")
        result = validate_design(
            design_id, seq, cif_path,
            method=args.method,
            work_dir=args.scratch,
            af2_server=args.af2_server,
            api_key=args.api_key,
            rmsd_threshold=args.rmsd_threshold,
            pae_threshold=args.pae_threshold,
        )
        rows.append(result._asdict())
        status = "✓" if result.flag_ok else "✗"
        print(f"  {status} {design_id}  rmsd={result.af2_rmsd:.2f}  pae={result.af2_pae:.2f}  plddt={result.af2_plddt:.2f}  {result.error}")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(args.out, index=False)
    n_ok = df_out["flag_ok"].sum()
    print(f"\nWrote {args.out}  ({n_ok}/{len(rows)} designs passed AF validation)")


if __name__ == "__main__":
    main()