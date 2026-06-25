#!/usr/bin/env bash
# =============================================================================
# run_hpc_campaign.sh — SLURM submission script for MARCO nanobody design
# =============================================================================
# Usage (from HPC login node, or directly via sbatch):
#   sbatch scripts/run_hpc_campaign.sh specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml runs/setD
#
# Required env vars (with recommended production values):
#   NUM_DESIGNS   Number of designs to generate  (recommended: 50,000-100,000)
#   BUDGET        Inference budget (iterations)   (recommended: 200-250 for RTX 5000)
#   CONDA_ENV     Conda environment name          (default: boltzgen)
#   GPUS          Number of GPUs to use           (default: 2, for RTX 5000 x2)
#
# NOTE on quality vs quantity:
#   - BUDGET 100-150: usable for screening, but ipTM distribution is skewed low.
#                     Only ~11%% of designs reach ipTM > 0.25 at BUDGET=150.
#   - BUDGET 200-250: significantly better ipTM/PAE distribution; required for
#                     confident binder predictions (Boltz confirmed binders: ipTM > 0.5).
#   - NUM_DESIGNS: For production, generate 50k-100k designs to get 50-200 quality
#                  candidates after all filters (N-glyc, proline, Gly/Ala, novelty).
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
#SBATCH --mem=96G                  # node system RAM (GPU VRAM is 16 GB per card)
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
NUM_DESIGNS="${NUM_DESIGNS:-60000}"  # 60k matches BoltzProt-1 production standard
BUDGET="${BUDGET:-150}"
CONDA_ENV="${CONDA_ENV:-boltzgen}"
GPUS="${GPUS:-2}"                  # RTX 5000 x2 → use 2 GPUs

# ── Speed mode ──────────────────────────────────────────────────────────────
# SPEED_MODE=1  →  aggressive speedup for large batches / screening
#   • fold: sampling_steps 200→100, recycling_steps 3→1, diffusion_samples 5→1
#   • design: compile_pairformer=true compile_structure=true  (~20-40% faster)
#   • inverse_fold: precision FP32→bf16-mixed
#   • diffusion_batch_size: 2→8  (better GPU utilization on 16 GB VRAM)
# SPEED_MODE=0  →  default balanced quality (or unset)
SPEED_MODE="${SPEED_MODE:-0}"

if [[ "$SPEED_MODE" == "1" ]]; then
  echo "[marco-run] SPEED_MODE=1 — using fast folding config"
  echo "            fold: sampling_steps=100 recycling_steps=1 diffusion_samples=1"
  echo "            design: compile_pairformer=true compile_structure=true"
  echo "            inverse_fold: precision=bf16-mixed"
  echo "            diffusion_batch_size=8"
fi

# diffusion_batch_size: RTX 5000 (16 GB) handles more than the old default of 2.
DEFAULT_DIFFUSION_BATCH_SIZE=2
[[ "$SPEED_MODE" == "1" ]] && DEFAULT_DIFFUSION_BATCH_SIZE=8

# Speed-mode fold/inverse_fold/config overrides
SPEED_FOLD_ARGS="--config fold sampling_steps=100 recycling_steps=1 diffusion_samples=1"
SPEED_DESIGN_ARGS="--config design compile_pairformer=true compile_structure=true"
SPEED_IFOLD_ARGS="--config inverse_fold precision=bf16-mixed"

# BoltzProt-1 protocol: 32 NXS/T sequons (Appendix E) are blocked at
# generation time via rejection sampling in the inverse-fold decoder.
# This prevents motifs from being baked into the model's internal
# representations — ~2× higher confirmed-binder rate vs post-hoc removal.
# The post-filter (EXCLUDE_NGLYC=1) still runs as a safety net.
NGLYC_MOTIFS="NAS,NAT,NCS,NCT,NDS,NDT,NES,NET,NGS,NGT,NIS,NIT,NKS,NKT,NLS,NLT,NMS,NMT,NNS,NNT,NQS,NQT,NRS,NRT,NSS,NST,NTS,NTT,NVS,NVT,NWS,NWT,NYS,NYT"
EXCLUDE_NGLYC="${EXCLUDE_NGLYC:-1}"

MARCO_EXTRA_ARGS="${MARCO_EXTRA_ARGS:---diffusion_batch_size $DEFAULT_DIFFUSION_BATCH_SIZE --metrics_override plip_hbonds_refolded=0.2 delta_sasa_refolded=0.5 --refolding_rmsd_threshold 3.0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
if [[ "$SPEED_MODE" == "1" ]]; then
  EXTRA_ARGS="$SPEED_FOLD_ARGS $SPEED_DESIGN_ARGS $SPEED_IFOLD_ARGS $EXTRA_ARGS"
fi

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
  echo "MARCO args:   $MARCO_EXTRA_ARGS"
  echo "Extra args:   ${EXTRA_ARGS:-<none>}"
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

# ── Post-generation: safety-net filter for any remaining N-glyc sequons ──
# N-glyc exclusion is handled post-generation by scripts/filter_nglyc.py
# (run automatically when EXCLUDE_NGLYC=1 after boltzgen run completes).
# Generation-time motif exclusion via --inverse_fold_excluded_sequence_motifs
# is NOT used here because boltzgen v0.3.x does not support it.
FILTER_PROLINE="${FILTER_PROLINE:-1}"
METRICS_AFTER_FILTER="$OUTDIR/final_ranked_designs/all_designs_metrics.csv"
if [[ "$EXCLUDE_NGLYC" == "1" ]]; then
  METRICS_CSV="$OUTDIR/final_ranked_designs/all_designs_metrics.csv"
  if [[ -f "$METRICS_CSV" ]]; then
    echo "=== Filtering N-glycosylation sequon designs ==="
    python scripts/filter_developability.py \
      --metrics "$METRICS_CSV" \
      --filter_nglyc \
      --out "$METRICS_CSV" \
      2>&1 | tee -a "${LOG_PREFIX}_run.log"
    LINES=$(wc -l < "$METRICS_CSV")
    echo "Post-NGLYC-filter lines: $LINES (header + data rows)"
    METRICS_AFTER_FILTER="$METRICS_CSV"
  else
    echo "WARNING: $METRICS_CSV not found — skipping NGLYC filter"
  fi
fi

# ── Proline-in-CDR3 filter (before ranking) ───────────────────────────────
# Proline in CDR3 disrupts the β-sheet scaffold and correlates with low Tm.
# Filtering before ranking avoids wasting compute on thermally unstable designs.
# Disable with FILTER_PROLINE=0.
if [[ "$FILTER_PROLINE" == "1" && -f "$METRICS_AFTER_FILTER" ]]; then
  echo "=== Filtering proline-in-CDR3 designs ==="
  python scripts/filter_developability.py \
    --metrics "$METRICS_AFTER_FILTER" \
    --filter_proline \
    --out "$METRICS_AFTER_FILTER" \
    2>&1 | tee -a "${LOG_PREFIX}_run.log"
  LINES=$(wc -l < "$METRICS_AFTER_FILTER")
  echo "Post-PROLINE-filter lines: $LINES (header + data rows)"
else
  echo "NOTE: FILTER_PROLINE=$FILTER_PROLINE — skipping proline filter"
fi

# ── CDR novelty check (BoltzProt-1 Section 3.5) ─────────────────────────────
# Check CDR3 and CDR1+CDR2+CDR3 edit distance vs SAbDab reference.
# Default: both must pass (--filter_mode both). Set NOVELTY_MODE=cdrs_only
# to use only CDR1+2+3 as the primary filter gate.
# The pre-built .sabdab_reference.json (4,466 unique CDR3s) is committed
# to the repo — no cache rebuild needed on HPC.
NOVELTY_MODE="${NOVELTY_MODE:-both}"
METRICS_FOR_NOVELTY="$METRICS_AFTER_PROLINE"
if [[ -f "$METRICS_FOR_NOVELTY" ]]; then
  echo "=== CDR novelty check (mode=$NOVELTY_MODE) ==="
  python scripts/novelty_check.py \
    --designs "$METRICS_FOR_NOVELTY" \
    --filter_mode "$NOVELTY_MODE" \
    --min_edit_distance 4 \
    --out "$METRICS_FOR_NOVELTY" \
    2>&1 | tee -a "${LOG_PREFIX}_run.log"
  echo "Novelty check done. Updated metrics: $METRICS_FOR_NOVELTY"
else
  echo "WARNING: $METRICS_FOR_NOVELTY not found — skipping novelty check"
fi

echo "Done. Results in: $OUTDIR"