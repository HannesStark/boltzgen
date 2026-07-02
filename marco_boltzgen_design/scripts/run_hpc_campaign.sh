#!/usr/bin/env bash
# =============================================================================
# run_hpc_campaign.sh — SLURM submission for MARCO nanobody design on RTX 5000×2
#
# USAGE (from HPC login node):
#   SIF=/path/to/ubuntu24_boltzgen.sif sbatch scripts/run_hpc_campaign.sh \
#     specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml runs/setD_rtx
#
#   SIF=/path/to/ubuntu24_boltzgen.sif SPEED_MODE=1 NUM_DESIGNS=60000 BUDGET=150 \
#     sbatch scripts/run_hpc_campaign.sh \
#     specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml runs/setD_screen
#
# ENVIRONMENT VARIABLES:
#   SIF                  Path to the ubuntu24_boltzgen.sif container  [REQUIRED]
#   NUM_DESIGNS          Designs per job                               (default: 60000)
#   BUDGET               Inference steps  (RTX 5000: 150–200 recommended) (default: 150)
#   GPUS                 GPU count (RTX 5000 ×2 → 2)                   (default: 2)
#   SPEED_MODE           1=fast, 0=quality                             (default: 0)
#   EXCLUDE_NGLYC        Auto-filter N-glyc sequons                    (default: 1)
#   FILTER_PROLINE       Auto-filter proline-in-CDR3                  (default: 1)
#
# INSIDE THE CONTAINER:
#   Python venv at /opt/venv (pip-installed boltzgen) is auto-activated.
#
# OUTPUT:
#   runs/<name>/final_ranked_designs/all_designs_metrics.csv
#   runs/<name>/final_ranked_designs/*.cif
# =============================================================================

#SBATCH --job-name=marco_vhh_design
#SBATCH --gres=gpu:2               # 2× RTX 5000 on same node
#SBATCH --cpus-per-task=16         # 8 CPU threads per GPU
#SBATCH --mem=96G                  # RTX 5000: 16 GB VRAM per card
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

# ── Validate SIF ─────────────────────────────────────────────────────────────
if [[ -z "${SIF:-}" ]]; then
  echo "ERROR: SIF environment variable is not set." >&2
  echo "Please provide the path to ubuntu24_boltzgen.sif:" >&2
  echo "  export SIF=/path/to/ubuntu24_boltzgen.sif" >&2
  echo "  sbatch scripts/run_hpc_campaign.sh ..." >&2
  exit 1
fi

if [[ ! -f "$SIF" ]]; then
  echo "ERROR: SIF not found: $SIF" >&2
  exit 1
fi

# ── Args ─────────────────────────────────────────────────────────────────────
SPEC="${1:-specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml}"
OUTDIR="${2:-runs/slurm_${SLURM_JOB_ID:-manual}}"
NUM_DESIGNS="${NUM_DESIGNS:-60000}"
BUDGET="${BUDGET:-150}"
GPUS="${GPUS:-2}"                  # RTX 5000 ×2
SPEED_MODE="${SPEED_MODE:-0}"

# diffusion_batch_size: RTX 5000 (16 GB) fits 8 in SPEED_MODE
DIFFUSION_BATCH_SIZE="${DIFFUSION_BATCH_SIZE:-2}"
[[ "$SPEED_MODE" == "1" && "${DIFFUSION_BATCH_SIZE:-2}" == "2" ]] && DIFFUSION_BATCH_SIZE=8

# ── Validate spec ──────────────────────────────────────────────────────────────
if [[ ! -f "$SPEC" ]]; then
  echo "ERROR: spec not found: $SPEC" >&2
  exit 1
fi

SPEC_NAME="$(basename "$SPEC" .yaml)"
LOG_PREFIX="logs/${SPEC_NAME}_${SLURM_JOB_ID:-local}"

mkdir -p logs "$OUTDIR"

# ── Log environment ────────────────────────────────────────────────────────────
{
  echo "===== $(date) — MARCO VHH Design Job (RTX 5000 × $GPUS) ====="
  echo "Spec:                 $SPEC"
  echo "Output dir:           $OUTDIR"
  echo "Num designs:          $NUM_DESIGNS"
  echo "Budget:               $BUDGET"
  echo "GPUs:                 $GPUS × RTX 5000 16GB"
  echo "diffusion_batch_size: $DIFFUSION_BATCH_SIZE"
  echo "SPEED_MODE:           $SPEED_MODE"
  echo "SIF:                  $SIF"
  echo "Job ID:               ${SLURM_JOB_ID:-local}"
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || echo "(nvidia-smi not available)"
} >> "${LOG_PREFIX}_start.log"

# ════════════════════════════════════════════════════════════════════════════
# Run the FULL pipeline inside the Apptainer container.
# ════════════════════════════════════════════════════════════════════════════
apptainer exec --nv \
  --bind "$PWD" \
  --bind "$(dirname "$SIF"):/opt/sif:ro" \
  "$SIF" \
  bash -c '
set -euo pipefail

# ── Thread limits (MUST be before activating venv) ───────────────────────────
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ── Activate Python venv inside the container ─────────────────────────────────
if [[ -f /opt/venv/bin/activate ]]; then
  source /opt/venv/bin/activate
else
  echo "ERROR: /opt/venv/bin/activate not found in container" >&2
  exit 1
fi

PROJECT_DIR="$(cat /proc/1/cwd 2>/dev/null || echo /project)"
cd "$PROJECT_DIR"

SPEC="${SPEC:-specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml}"
OUTDIR="${OUTDIR:-runs/slurm}"
NUM_DESIGNS="${NUM_DESIGNS:-60000}"
BUDGET="${BUDGET:-150}"
GPUS="${GPUS:-2}"
DIFFUSION_BATCH_SIZE="${DIFFUSION_BATCH_SIZE:-2}"
SPEED_MODE="${SPEED_MODE:-0}"

SPEC_NAME="$(basename "$SPEC" .yaml)"
LOG_PREFIX="logs/${SPEC_NAME}_${SLURM_JOB_ID:-local}"
mkdir -p logs "$OUTDIR"

MARCO_EXTRA_ARGS="--diffusion_batch_size $DIFFUSION_BATCH_SIZE \
  --metrics_override plip_hbonds_refolded=0.2 delta_sasa_refolded=0.5 \
  --refolding_rmsd_threshold 3.0"

SPEED_FOLD_ARGS=""
SPEED_DESIGN_ARGS=""
SPEED_IFOLD_ARGS=""
if [[ "$SPEED_MODE" == "1" ]]; then
  SPEED_FOLD_ARGS="--config fold sampling_steps=100 recycling_steps=1 diffusion_samples=1"
  SPEED_DESIGN_ARGS="--config design compile_pairformer=true compile_structure=true"
  SPEED_IFOLD_ARGS="--config inverse_fold precision=bf16-mixed"
fi

EXTRA_ARGS="${EXTRA_ARGS:-} $SPEED_FOLD_ARGS $SPEED_DESIGN_ARGS $SPEED_IFOLD_ARGS"

# ══ STEP 1: BoltzGen generation ══════════════════════════════════════════════
echo "===== $(date) — Starting boltzgen (devices=$GPUS, batch=$DIFFUSION_BATCH_SIZE) =====" | tee -a "${LOG_PREFIX}_run.log"

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
echo "===== $(date) — boltzgen done (exit=$RUN_EXIT) =====" | tee -a "${LOG_PREFIX}_run.log"

if [[ $RUN_EXIT -ne 0 ]]; then
  echo "ERROR: boltzgen run failed with exit code $RUN_EXIT" >&2
  exit $RUN_EXIT
fi

METRICS_CSV="$OUTDIR/final_ranked_designs/all_designs_metrics.csv"
METRICS="$METRICS_CSV"

# ══ STEP 2: N-glyc filter ═══════════════════════════════════════════════════
EXCLUDE_NGLYC="${EXCLUDE_NGLYC:-1}"
if [[ "$EXCLUDE_NGLYC" == "1" && -f "$METRICS_CSV" ]]; then
  echo "=== N-glyc filter ===" | tee -a "${LOG_PREFIX}_run.log"
  python scripts/filter_nglyc.py \
    --metrics "$METRICS_CSV" \
    --out "$METRICS_CSV" \
    2>&1 | tee -a "${LOG_PREFIX}_run.log"
fi

# ══ STEP 3: Developability filter (N-glyc + proline) ════════════════════════
FILTER_PROLINE="${FILTER_PROLINE:-1}"
METRICS_AFTER_FILTER="$METRICS_CSV"
if [[ -f "$METRICS_AFTER_FILTER" ]]; then
  python scripts/filter_developability.py \
    --metrics "$METRICS_AFTER_FILTER" \
    --filter_nglyc \
    --filter_proline \
    --filter_mode hard \
    --out "$METRICS_AFTER_FILTER" \
    2>&1 | tee -a "${LOG_PREFIX}_run.log"
fi

# ══ STEP 4: Novelty check ═══════════════════════════════════════════════════
NOVELTY_MODE="${NOVELTY_MODE:-both}"
if [[ -f "$METRICS_AFTER_FILTER" ]]; then
  echo "=== CDR novelty check (mode=$NOVELTY_MODE) ===" | tee -a "${LOG_PREFIX}_run.log"
  python scripts/novelty_check.py \
    --designs "$METRICS_AFTER_FILTER" \
    --filter_mode "$NOVELTY_MODE" \
    --min_edit_distance 4 \
    --out "$METRICS_AFTER_FILTER" \
    2>&1 | tee -a "${LOG_PREFIX}_run.log"
fi

echo "Pipeline complete. Results in: $OUTDIR"
'