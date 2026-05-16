#!/usr/bin/env bash
# =============================================================================
# run_hpc_campaign.sh — SLURM submission script for MARCO nanobody design
# =============================================================================
# Usage (from HPC login node, or directly via sbatch):
#   sbatch scripts/run_hpc_campaign.sh specs/mouse_marco_nanobody_hotspot.yaml runs/mouse_vhh_batch1
#
# Required env vars (with defaults):
#   NUM_DESIGNS   Number of designs to generate  (default: 2000)
#   BUDGET        Inference budget (iterations)   (default: 150)
#   CONDA_ENV     Conda environment name          (default: boltzgen)
#   GPUS          Number of GPUs to use           (default: 2, for RTX 5000 x2)
#
# GPU strategy: With 2x RTX 5000 (16 GB each), use --devices $GPUS so
# BoltzGen splits the batch across both GPUs within a single job. The SLURM
# job requests both GPUs at once so they are reserved together on the same node.
#
# Output:
#   runs/<name>/final_ranked_designs/all_designs_metrics.csv
#   runs/<name>/final_ranked_designs/*.cif
# =============================================================================

#SBATCH --job-name=marco_vhh_design
#SBATCH --gres=gpu:2               # 2x RTX 5000 on same node
#SBATCH --cpus-per-task=16         # 8 CPU threads per GPU
#SBATCH --mem=96G                  # shared memory pool for 2 GPUs (48G each)
#SBATCH --time=96:00:00            # RTX 5000 is slower than A100; give more time
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu            # adjust to your cluster (gpu, gambit, dGX, etc.)
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$(whoami)@zzu.edu.cn

set -euo pipefail

# ── Resolve project root ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# ── Args ─────────────────────────────────────────────────────────────────────
SPEC="${1:-specs/crossreactive_marco_nanobody_hotspot.yaml}"
OUTDIR="${2:-runs/slurm_${SLURM_JOB_ID:-manual}}"
NUM_DESIGNS="${NUM_DESIGNS:-2000}"
BUDGET="${BUDGET:-150}"
CONDA_ENV="${CONDA_ENV:-boltzgen}"
GPUS="${GPUS:-2}"                  # RTX 5000 x2 → use 2 GPUs

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
  echo "GPUs:         $GPUS × RTX 5000"
  echo "Conda env:    $CONDA_ENV"
  echo "Job ID:       ${SLURM_JOB_ID:-local}"
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || echo "(nvidia-smi not available)"
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
# Steps:
#   [1] design          Generate novel CDR sequences attached to scaffolds
#   [2] inverse_folding Score/filter by scaffold structure recovery
#   [3] folding         Predict full binder + target complex
#   [4] analysis        Compute pLDDT, ipTM, PAE, interface metrics
#   [5] filtering       Apply confidence/diversity filters
#
# --devices $GPUS:  Distribute the batch across all available GPUs (RTX 5000 x2)
# --reuse:          Skip already-designed structures if re-running after timeout/OOM
echo "===== $(date) — Starting boltzgen run (devices=$GPUS) =====" | tee -a "${LOG_PREFIX}_run.log"

boltzgen run "$SPEC" \
  --output "$OUTDIR" \
  --protocol nanobody-anything \
  --num_designs "$NUM_DESIGNS" \
  --budget "$BUDGET" \
  --devices "$GPUS" \
  --reuse \
  2>&1 | tee -a "${LOG_PREFIX}_run.log"

RUN_EXIT=${PIPESTATUS[0]}

{
  echo "===== $(date) — boltzgen run finished (exit code: $RUN_EXIT) ====="
  if [[ -f "$OUTDIR/final_ranked_designs/all_designs_metrics.csv" ]]; then
    LINES=$(wc -l < "$OUTDIR/final_ranked_designs/all_designs_metrics.csv")
    echo "Metrics CSV lines: $LINES (header + data rows)"
  else
    echo "WARNING: final metrics CSV not found — job may have been killed early"
  fi
} >> "${LOG_PREFIX}_done.log"

if [[ $RUN_EXIT -ne 0 ]]; then
  echo "ERROR: boltzgen run failed with exit code $RUN_EXIT" >&2
  exit $RUN_EXIT
fi

echo "Done. Results in: $OUTDIR"