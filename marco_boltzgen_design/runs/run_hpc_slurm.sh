#!/usr/bin/env bash
#SBATCH --job-name=boltzgen_marco
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

SPEC="${1:-specs/crossreactive_conserved_surface.yaml}"
PROTOCOL="${2:-protein-anything}"
OUTDIR="${3:-runs/slurm_${SLURM_JOB_ID:-manual}}"
NUM_DESIGNS="${NUM_DESIGNS:-1000}"
BUDGET="${BUDGET:-100}"

mkdir -p logs
nvidia-smi || true

CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"
if [[ -n "$CONDA_ENV_NAME" ]]; then
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV_NAME"
  else
    echo "CONDA_ENV_NAME was set but 'conda' is not available on PATH" >&2
    exit 1
  fi
fi

boltzgen run "$SPEC" \
  --output "$OUTDIR" \
  --protocol "$PROTOCOL" \
  --num_designs "$NUM_DESIGNS" \
  --budget "$BUDGET" \
  --devices 1 \
  --reuse
