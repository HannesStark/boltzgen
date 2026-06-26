#!/usr/bin/env bash
# =============================================================================
# run_a100hpc_campaign.sh — SLURM submission for MARCO nanobody design on A100×4
# =============================================================================
# Usage (from HPC login node, or directly via sbatch):
#   A100_MODE=1 NUM_DESIGNS=60000 BUDGET=200 sbatch scripts/run_a100hpc_campaign.sh \
#     specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml runs/setD_a100
#
# Or use the built-in A100_MODE which sets all defaults at once:
#   sbatch scripts/run_a100hpc_campaign.sh \
#     specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml runs/setD_a100
#
# Key environment variables:
#   A100_MODE           Set to 1 to auto-apply all A100-optimal defaults (recommended)
#   NUM_DESIGNS         Designs per job  (default: 60000)
#   BUDGET              Inference steps  (default: 200 — good for A100)
#   GPUS                GPUs per job     (default: 4)
#   DIFFUSION_BATCH_SIZE Batch size per GPU step  (default: 16, or 32 in SPEED_MODE)
#   SPEED_MODE          Set to 1 for screening-quality fast runs
#   CONDA_ENV           Conda environment  (default: boltzgen)
#   EXCLUDE_NGLYC       Auto-filter N-glyc sequons  (default: 1)
#   FILTER_PROLINE      Auto-filter proline-in-CDR3  (default: 1)
#
# GPU notes:
#   A100 80GB has ~5× the VRAM of RTX 5000 16GB. diffusion_batch_size and
#   recycling_steps are the main levers to fill the larger memory budget.
#
# Output:
#   runs/<name>/final_ranked_designs/all_designs_metrics.csv
#   runs/<name>/final_ranked_designs/*.cif
# =============================================================================

#SBATCH --job-name=marco_vhh_a100
#SBATCH --gres=gpu:4               # 4× A100 80GB on same node
#SBATCH --cpus-per-task=32         # 8 CPU threads per GPU
#SBATCH --mem=256G                 # A100 nodes typically have 256–512 GB system RAM
#SBATCH --time=72:00:00            # A100 ×4 at BUDGET=200 finishes well within 72h
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

# ── A100 defaults ──────────────────────────────────────────────────────────
# Set A100_MODE=1 to auto-apply these. Individual vars still override.
A100_MODE="${A100_MODE:-1}"
if [[ "$A100_MODE" == "1" ]]; then
  export GPUS="${GPUS:-4}"
  export BUDGET="${BUDGET:-200}"
  export DIFFUSION_BATCH_SIZE="${DIFFUSION_BATCH_SIZE:-16}"
  export SPEED_MODE="${SPEED_MODE:-0}"
  export EXCLUDE_NGLYC="${EXCLUDE_NGLYC:-1}"
  export FILTER_PROLINE="${FILTER_PROLINE:-1}"
  echo "[A100_MODE] Applied A100-optimal defaults: GPUS=$GPUS BUDGET=$BUDGET DIFFUSION_BATCH_SIZE=$DIFFUSION_BATCH_SIZE SPEED_MODE=$SPEED_MODE"
fi

# ── Args ─────────────────────────────────────────────────────────────────────
SPEC="${1:-specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml}"
OUTDIR="${2:-runs/slurm_${SLURM_JOB_ID:-manual}}"
NUM_DESIGNS="${NUM_DESIGNS:-60000}"
BUDGET="${BUDGET:-200}"
CONDA_ENV="${CONDA_ENV:-boltzgen}"
GPUS="${GPUS:-4}"
DIFFUSION_BATCH_SIZE="${DIFFUSION_BATCH_SIZE:-16}"   # 16 is safe default for A100 80GB
SPEED_MODE="${SPEED_MODE:-0}"

# ── Validate ─────────────────────────────────────────────────────────────────
if [[ ! -f "$SPEC" ]]; then
  echo "ERROR: spec not found: $SPEC" >&2
  echo "Usage: A100_MODE=1 sbatch $0 [SPEC.yaml] [output_dir]" >&2
  exit 1
fi

SPEC_NAME="$(basename "$SPEC" .yaml)"
LOG_PREFIX="logs/${SPEC_NAME}_${SLURM_JOB_ID:-local}"

mkdir -p logs "$OUTDIR"

# ── Speed mode ──────────────────────────────────────────────────────────────
# SPEED_MODE=1 →  aggressive speedup for screening / large batches
#   fold: sampling_steps 200→100, recycling_steps 3→1, diffusion_samples 5→1
#   design: compile_pairformer=true compile_structure=true  (~20-40% faster)
#   inverse_fold: precision FP32→bf16-mixed
#   diffusion_batch_size: 16→32  (better A100 80GB utilization)
if [[ "$SPEED_MODE" == "1" ]]; then
  echo "[A100] SPEED_MODE=1 — using fast config"
  echo "       fold: recycling_steps=1 compile_structure=true"
  echo "       design: compile_pairformer=true compile_structure=true"
  echo "       inverse_fold: precision=bf16-mixed"
  echo "       diffusion_batch_size=$DIFFUSION_BATCH_SIZE"
  SPEED_FOLD_ARGS="--config fold recycling_steps=1 compile_structure=true"
  SPEED_DESIGN_ARGS="--config design compile_pairformer=true compile_structure=true"
  SPEED_IFOLD_ARGS="--config inverse_fold precision=bf16-mixed"
  # Override DIFFUSION_BATCH_SIZE to a larger value in speed mode if not already set
  [[ "${DIFFUSION_BATCH_SIZE:-16}" == "16" ]] && DIFFUSION_BATCH_SIZE=32
else
  # Quality mode: still compile the structure module for ~20-40% fold speedup
  SPEED_FOLD_ARGS="--config fold compile_structure=true"
  SPEED_DESIGN_ARGS=""
  SPEED_IFOLD_ARGS=""
fi

# ── Log environment ──────────────────────────────────────────────────────────
{
  echo "===== $(date) — MARCO VHH Design Job (A100 × $GPUS) ====="
  echo "Spec:                $SPEC"
  echo "Output dir:          $OUTDIR"
  echo "Num designs:         $NUM_DESIGNS"
  echo "Budget:              $BUDGET"
  echo "GPUs:                $GPUS × A100 80GB"
  echo "diffusion_batch_size: $DIFFUSION_BATCH_SIZE"
  echo "SPEED_MODE:          $SPEED_MODE"
  echo "EXCLUDE_NGLYC:       $EXCLUDE_NGLYC"
  echo "FILTER_PROLINE:      $FILTER_PROLINE"
  echo "Conda env:           $CONDA_ENV"
  echo "Job ID:              ${SLURM_JOB_ID:-local}"
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || echo "(nvidia-smi not available)"
} >> "${LOG_PREFIX}_start.log"

# ── Thread limits ────────────────────────────────────────────────────────────
# OpenBLAS, OpenMP, MKL each try to spawn many threads by default.
# On shared HPC nodes this can exhaust RLIMIT_NPROC and cause:
#   "pthread_create failed: Resource temporarily unavailable"
# Cap them to 1 thread each — all heavy GPU work is done by BoltzGen
# via PyTorch, which manages its own thread pool independently.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ── Activate conda ────────────────────────────────────────────────────────────
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
else
  echo "ERROR: conda not found on PATH" >&2
  exit 1
fi

# ── MARCO extra args ──────────────────────────────────────────────────────────
# diffusion_batch_size: 16 for A100 (safe default for 80GB VRAM)
# Increase to 24–32 in SPEED_MODE if VRAM allows (run a quick benchmark to tune)
MARCO_EXTRA_ARGS="--diffusion_batch_size $DIFFUSION_BATCH_SIZE \
  --metrics_override plip_hbonds_refolded=0.2 delta_sasa_refolded=0.5 \
  --refolding_rmsd_threshold 3.0"

EXTRA_ARGS="${EXTRA_ARGS:-} $SPEED_FOLD_ARGS $SPEED_DESIGN_ARGS $SPEED_IFOLD_ARGS"
EXTRA_ARGS="${EXTRA_ARGS//  / }"  # collapse any double spaces

# ── Run BoltzGen ─────────────────────────────────────────────────────────────
echo "===== $(date) — Starting boltzgen run (devices=$GPUS, batch=$DIFFUSION_BATCH_SIZE) =====" | tee -a "${LOG_PREFIX}_run.log"

boltzgen run "$SPEC" \
  --output "$OUTDIR" \
  --protocol nanobody-anything \
  --num_designs "$NUM_DESIGNS" \
  --budget "$BUDGET" \
  --devices "$GPUS" \
  --reuse \
  $MARCO_EXTRA_ARGS \
  $EXTRA_ARGS \
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

# ── Post-generation: N-glyc sequon filter ─────────────────────────────────────
METRICS_CSV="$OUTDIR/final_ranked_designs/all_designs_metrics.csv"
METRICS_AFTER_FILTER="$METRICS_CSV"

if [[ "$EXCLUDE_NGLYC" == "1" && -f "$METRICS_CSV" ]]; then
  echo "=== Filtering N-glycosylation sequon designs ==="
  python scripts/filter_developability.py \
    --metrics "$METRICS_CSV" \
    --filter_nglyc \
    --out "$METRICS_CSV" \
    2>&1 | tee -a "${LOG_PREFIX}_run.log"
  LINES=$(wc -l < "$METRICS_CSV")
  echo "Post-NGLYC-filter lines: $LINES"
  METRICS_AFTER_FILTER="$METRICS_CSV"
fi

# ── Proline-in-CDR3 filter ───────────────────────────────────────────────────
FILTER_PROLINE="${FILTER_PROLINE:-1}"
if [[ "$FILTER_PROLINE" == "1" && -f "$METRICS_AFTER_FILTER" ]]; then
  echo "=== Filtering proline-in-CDR3 designs ==="
  python scripts/filter_developability.py \
    --metrics "$METRICS_AFTER_FILTER" \
    --filter_proline \
    --out "$METRICS_AFTER_FILTER" \
    2>&1 | tee -a "${LOG_PREFIX}_run.log"
  LINES=$(wc -l < "$METRICS_AFTER_FILTER")
  echo "Post-PROLINE-filter lines: $LINES"
fi

# ── CDR novelty check ───────────────────────────────────────────────────────
NOVELTY_MODE="${NOVELTY_MODE:-both}"
METRICS_FOR_NOVELTY="$METRICS_AFTER_FILTER"
if [[ -f "$METRICS_FOR_NOVELTY" ]]; then
  echo "=== CDR novelty check (mode=$NOVELTY_MODE) ==="
  python scripts/novelty_check.py \
    --designs "$METRICS_FOR_NOVELTY" \
    --filter_mode "$NOVELTY_MODE" \
    --min_edit_distance 4 \
    --out "$METRICS_FOR_NOVELTY" \
    2>&1 | tee -a "${LOG_PREFIX}_run.log"
  echo "Novelty check done."
else
  echo "WARNING: $METRICS_FOR_NOVELTY not found — skipping novelty check"
fi

echo "Done. Results in: $OUTDIR"