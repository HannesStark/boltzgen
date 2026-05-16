#!/usr/bin/env bash
# =============================================================================
# run_hpc_campaign.sh — SLURM submission script for MARCO nanobody design
# =============================================================================
# Usage (from HPC login node or via `sbatch`):
#   sbatch runs/run_hpc_campaign.sh specs/mouse_marco_nanobody_hotspot.yaml runs/mouse_vhh_prod
#
# Required env vars (with defaults):
#   NUM_DESIGNS   Number of designs to generate  (default: 1000)
#   BUDGET        Inference budget (iterations)  (default: 200)
#   CONDA_ENV     Conda environment name         (default: boltzgen)
#
# Output:
#   runs/<name>/final_ranked_designs/all_designs_metrics.csv
#   runs/<name>/final_ranked_designs/*.cif
# =============================================================================

#SBATCH --job-name=marco_vhh_design
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu          # adjust to your cluster (e.g., gpu, gambit, dGX)
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$(whoami)@zzu.edu.cn

set -euo pipefail

# ── Resolve project root (assumes script lives at boltzgen/marco_boltzgen_design/scripts/) ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# ── Args ─────────────────────────────────────────────────────────────────────
SPEC="${1:-specs/crossreactive_marco_nanobody_hotspot.yaml}"
OUTDIR="${2:-runs/slurm_${SLURM_JOB_ID:-manual}}"
NUM_DESIGNS="${NUM_DESIGNS:-1000}"
BUDGET="${BUDGET:-200}"
CONDA_ENV="${CONDA_ENV:-boltzgen}"

# ── Validate ─────────────────────────────────────────────────────────────────
if [[ ! -f "$SPEC" ]]; then
  echo "ERROR: spec not found: $SPEC" >&2
  echo "Usage: sbatch $0 [SPEC.yaml] [output_dir]" >&2
  exit 1
fi

SPEC_NAME="$(basename "$SPEC" .yaml)"
LOG_PREFIX="logs/${SPEC_NAME}_${SLURM_JOB_ID:-local}"

mkdir -p logs "$OUTDIR"

# ── Log environment ──────────────────────────────────────────────────────────
{
  echo "===== $(date) — MARCO VHH Design Job ====="
  echo "Spec:         $SPEC"
  echo "Output dir:   $OUTDIR"
  echo "Num designs:  $NUM_DESIGNS"
  echo "Budget:       $BUDGET"
  echo "Conda env:    $CONDA_ENV"
  echo "Job ID:       ${SLURM_JOB_ID:-local}"
  nvidia-smi || echo "(nvidia-smi not available)"
} >> "${LOG_PREFIX}_start.log"

# ── Activate conda ────────────────────────────────────────────────────────────
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
else
  echo "ERROR: conda not found on PATH" >&2
  exit 1
fi

# ── Run the full 5-step nanobody-anything pipeline ───────────────────────────
# Step 1: design       — generate CDR sequences attached to scaffolds
# Step 2: inverse_folding — score/filter by scaffold structure recovery
# Step 3: folding      — predict full binder + target complex
# Step 4: analysis     — compute pLDDT, ipTM, PAE, interface metrics
# Step 5: filtering    — apply confidence/diversity filters
#
# --reuse: skip already-designed structures (safe to re-run after OOM/timeout)
echo "===== $(date) — Starting boltzgen run =====" | tee -a "${LOG_PREFIX}_run.log"

boltzgen run "$SPEC" \
  --output "$OUTDIR" \
  --protocol nanobody-anything \
  --num_designs "$NUM_DESIGNS" \
  --budget "$BUDGET" \
  --devices 1 \
  --reuse \
  2>&1 | tee -a "${LOG_PREFIX}_run.log"

RUN_EXIT=${PIPESTATUS[0]}

{
  echo "===== $(date) — boltzgen run finished (exit code: $RUN_EXIT) ====="
  if [[ -f "$OUTDIR/final_ranked_designs/all_designs_metrics.csv" ]]; then
    LINES=$(wc -l < "$OUTDIR/final_ranked_designs/all_designs_metrics.csv")
    echo "Metrics CSV lines: $LINES"
  else
    echo "WARNING: final metrics CSV not found"
  fi
} >> "${LOG_PREFIX}_done.log"

if [[ $RUN_EXIT -ne 0 ]]; then
  echo "ERROR: boltzgen run failed with exit code $RUN_EXIT" >&2
  exit $RUN_EXIT
fi

echo "Done. Results in: $OUTDIR"