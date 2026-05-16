#!/usr/bin/env bash
# =============================================================================
# rank_and_validate.sh — Stage 3 (rank) + Stage 4 (AF2 validation) orchestrator
# =============================================================================
# Usage:
#   ./scripts/rank_and_validate.sh \
#     --metrics results/all_metrics.csv \
#     --human-conserved "A:423,A:425,A:432,A:461,A:467,A:469,A:489,A:500" \
#     --mouse-conserved "A:6,A:8,A:15,A:44,A:50,A:52,A:72,A:83" \
#     --top_n 50 \
#     --method colabfold
#
# Outputs:
#   results/ranked_candidates.csv
#   results/af_validation.csv
#   results/candidate_cifs/  (top-50 designs as CIFs for downstream use)
# =============================================================================
set -euo pipefail

METRICS=""
HUMAN_CONS="A:423,A:425,A:432,A:461,A:467,A:469,A:489,A:500"
MOUSE_CONS="A:6,A:8,A:15,A:44,A:50,A:52,A:72,A:83"
MAX_LEN=140
TOP_N=50
METHOD="colabfold"
OUT_RANK="results/ranked_candidates.csv"
OUT_AF="results/af_validation.csv"
OUT_CIFS="results/candidate_cifs"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --metrics)          METRICS="$2"; shift 2 ;;
    --human-conserved)  HUMAN_CONS="$2"; shift 2 ;;
    --mouse-conserved)  MOUSE_CONS="$2"; shift 2 ;;
    --max_len)          MAX_LEN="$2"; shift 2 ;;
    --top_n)            TOP_N="$2"; shift 2 ;;
    --method)           METHOD="$2"; shift 2 ;;
    --out_rank)         OUT_RANK="$2"; shift 2 ;;
    --out_af)           OUT_AF="$2"; shift 2 ;;
    *) echo "Unknown: $1"; exit 1 ;;
  esac
done

if [[ -z "$METRICS" ]]; then
  echo "Usage: $0 --metrics RESULTS_CSV [...]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p results "$OUT_CIFS"

# ── Stage 3: Rank & developability filter ─────────────────────────────────────
echo "=== Stage 3: Ranking designs ==="
python scripts/rank_designs.py \
  --metrics "$METRICS" \
  --human-conserved "$HUMAN_CONS" \
  --mouse-conserved "$MOUSE_CONS" \
  --max_len "$MAX_LEN" \
  --out "$OUT_RANK"

N_TOTAL=$(wc -l < "$OUT_RANK" | tr -d ' ')
echo "Ranked candidates: ${N_TOTAL:-?} total rows"

# ── Copy top-N CIFs to candidate_cifs/ ────────────────────────────────────────
echo "=== Copying top-$TOP_N design CIFs ==="
if [[ -f "$METRICS" ]]; then
  python3 -c "
import pandas as pd, shutil, pathlib
df = pd.read_csv('$OUT_RANK')
top = df.head($TOP_N)
cif_dir = None
for run_dir in pathlib.Path('runs').glob('*/final_ranked_designs'):
    for cif in run_dir.glob('*.cif'):
        base = cif.stem
        if base in top['design_id'].values:
            dst = pathlib.Path('$OUT_CIFS') / cif.name
            shutil.copy2(cif, dst)
            print(f'  copied {cif.name}')
"
fi

# ── Stage 4: AF2 validation ───────────────────────────────────────────────────
echo "=== Stage 4: AF2 backfold validation (method=$METHOD) ==="
if [[ -d "$OUT_CIFS" ]] && [[ $(ls "$OUT_CIFS"/*.cif 2>/dev/null | wc -l) -gt 0 ]]; then
  python scripts/validate_designs.py \
    --complexes "$OUT_CIFS" \
    --metrics "$OUT_RANK" \
    --top_n "$TOP_N" \
    --method "$METHOD" \
    --out "$OUT_AF"
else
  echo "WARNING: No CIFs found in $OUT_CIFS — skipping AF2 validation"
  echo "  (Run this step after copying CIFs from HPC: rsync -avz hpc:path/to/final_ranked_designs/ results/candidate_cifs/)"
fi

echo ""
echo "=== Pipeline complete ==="
echo "  Ranking:    $OUT_RANK"
echo "  AF2 result: $OUT_AF"
echo "  CIFs:       $OUT_CIFS/"