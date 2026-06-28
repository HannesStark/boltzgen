#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — Standalone single-node pipeline for MARCO nanobody design
# on a machine with 4× NVIDIA GPUs (A100 or similar).
#
# No SLURM required. Runs entirely in the foreground (or background with &).
# All 5 pipeline steps complete in one invocation.
#
# Usage:
#   # Full quality run
#   bash scripts/run_pipeline.sh \
#     specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml setD_prod
#
#   # Fast screening run
#   SPEED_MODE=1 NUM_DESIGNS=60000 BUDGET=150 \
#     bash scripts/run_pipeline.sh \
#     specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml setD_screen
#
# Environment variables (all optional):
#   SPEC          Path to YAML spec file  [default: first arg]
#   OUTNAME       Run name for output dir  [default: second arg, or derived from spec]
#   GPUS          GPU count                [default: 4]
#   NUM_DESIGNS   Number of designs        [default: 60000]
#   BUDGET        Inference steps          [default: 200]
#   SPEED_MODE    1=fast, 0=quality       [default: 0]
#   MIN_IPTM      ipTM threshold           [default: 0.25]
#   TOP_N         Top designs to keep      [default: 500]
# =============================================================================

set -euo pipefail

# ── Input args ───────────────────────────────────────────────────────────────
SPEC="${1:-specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml}"
OUTNAME="${2:-}"
if [[ -z "$OUTNAME" ]]; then
  # Derive from spec filename e.g. specs/foo_bar.yaml → foo_bar
  OUTNAME="$(basename "$SPEC" .yaml)"
fi

# ── Config ──────────────────────────────────────────────────────────────────
GPUS="${GPUS:-4}"
NUM_DESIGNS="${NUM_DESIGNS:-60000}"
BUDGET="${BUDGET:-200}"
SPEED_MODE="${SPEED_MODE:-0}"
MIN_IPTM="${MIN_IPTM:-0.25}"
TOP_N="${TOP_N:-500}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTDIR="$PROJECT_DIR/runs/${OUTNAME}"
LOGDIR="$PROJECT_DIR/logs"
LOGFILE="$LOGDIR/${OUTNAME}_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$OUTDIR" "$LOGDIR"

# ── Thread limits (MUST be set before conda activate) ───────────────────────
# Prevents RLIMIT_NPROC exhaustion from OpenBLAS/MKL thread spawning on
# shared-node HPC environments. Safe to set on any system.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ── Conda ───────────────────────────────────────────────────────────────────
eval "$(conda shell.bash hook)"
conda activate boltzgen

# ── Logging helper ──────────────────────────────────────────────────────────
log() {
  local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  echo "$msg" | tee -a "$LOGFILE"
}

# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — Generate designs (diffusion → inverse fold → refold → fold → affinity)
# ════════════════════════════════════════════════════════════════════════════
log "═══════════════════════════════════════════════════════════════════"
log "[Step 1] BoltzGen generation"
log "  spec         : $SPEC"
log "  outdir       : $OUTDIR"
log "  GPUS         : $GPUS"
log "  num_designs  : $NUM_DESIGNS"
log "  budget       : $BUDGET"
log "  speed_mode   : $SPEED_MODE"
log "═══════════════════════════════════════════════════════════════════"

# Speed-mode args: ~3× faster fold step, bf16 for design
if [[ "$SPEED_MODE" == "1" ]]; then
  log "[Step 1] SPEED_MODE=1 — using fast config"
  FOLD_ARGS="--config fold sampling_steps=100 recycling_steps=1 diffusion_samples=1 compile_structure=true"
  DESIGN_ARGS="--config design compile_pairformer=true compile_structure=true"
  IFOLD_ARGS="--config inverse_fold precision=bf16-mixed"
  EXTRA_ARGS="$FOLD_ARGS $DESIGN_ARGS $IFOLD_ARGS"
else
  log "[Step 1] QUALITY_MODE — using compile_structure=true only"
  EXTRA_ARGS="--config fold compile_structure=true"
fi

boltzgen run "$SPEC" \
  --output "$OUTDIR" \
  --protocol nanobody-anything \
  --num_designs $NUM_DESIGNS \
  --budget $BUDGET \
  --devices cuda \
  --reuse \
  $EXTRA_ARGS \
  2>&1 | tee -a "$LOGFILE"

METRICS="$OUTDIR/final_ranked_designs/all_designs_metrics.csv"

if [[ ! -f "$METRICS" ]]; then
  log "[Step 1] ERROR: metrics file not found at $METRICS"
  exit 1
fi

log "[Step 1] Generation complete."
log "  Raw designs  : $(wc -l < "$METRICS") lines"

# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — N-glycosylation sequon filter
# ════════════════════════════════════════════════════════════════════════════
log "═══════════════════════════════════════════════════════════════════"
log "[Step 2] N-glycosylation sequon filter"
log "═══════════════════════════════════════════════════════════════════"

python scripts/filter_nglyc.py \
  --metrics "$METRICS" \
  --out "$METRICS" \
  2>&1 | tee -a "$LOGFILE"

log "[Step 2] N-glyc filter complete."
log "  After filter : $(wc -l < "$METRICS") lines"

# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — Developability filter
# ════════════════════════════════════════════════════════════════════════════
log "═══════════════════════════════════════════════════════════════════"
log "[Step 3] Developability filter"
log "═══════════════════════════════════════════════════════════════════"

python scripts/filter_developability.py \
  --metrics "$METRICS" \
  --out "$METRICS" \
  --filter_nglyc \
  --filter_mode hard \
  2>&1 | tee -a "$LOGFILE"

log "[Step 3] Developability filter complete."
log "  After filter : $(wc -l < "$METRICS") lines"

# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — CDR novelty check against SAbDab
# ════════════════════════════════════════════════════════════════════════════
log "═══════════════════════════════════════════════════════════════════"
log "[Step 4] CDR novelty check (SAbDab, min_edit_distance=4)"
log "═══════════════════════════════════════════════════════════════════"

python scripts/novelty_check.py \
  --designs "$METRICS" \
  --filter_mode both \
  --min_edit_distance 4 \
  --out "$METRICS" \
  2>&1 | tee -a "$LOGFILE"

log "[Step 4] Novelty check complete."
log "  After filter : $(wc -l < "$METRICS") lines"

# ════════════════════════════════════════════════════════════════════════════
# STEP 5 — Rank and extract top candidates
# ════════════════════════════════════════════════════════════════════════════
log "═══════════════════════════════════════════════════════════════════"
log "[Step 5] Rank designs (min_iptm=$MIN_IPTM, top $TOP_N)"
log "═══════════════════════════════════════════════════════════════════"

TOP_CSV="$OUTDIR/top${TOP_N}_candidates.csv"

python scripts/rank_designs.py \
  --metrics "$METRICS" \
  --min_iptm $MIN_IPTM \
  --max_pae 15 \
  --max_gly_ala_frac 0.35 \
  --top_n $TOP_N \
  --out "$TOP_CSV" \
  2>&1 | tee -a "$LOGFILE"

# ── Summary ───────────────────────────────────────────────────────────────────
log "═══════════════════════════════════════════════════════════════════"
log "PIPELINE COMPLETE"
log "═══════════════════════════════════════════════════════════════════"
log "  Output dir   : $OUTDIR"
log "  Full metrics : $METRICS"
log "  Top candidates: $TOP_CSV"
log "  Log file     : $LOGFILE"

# Show top 10 by final_score if rank_designs succeeded
if [[ -f "$TOP_CSV" ]]; then
  log ""
  log "─── Top 10 candidates by final_score ───"
  head -11 "$TOP_CSV" | awk -F',' 'NR==1{next} {printf "  %-6s ipTM=%.3f pAE=%.1f score=%.4f  %s\n", $1, $4, $5, $NF, $2}' | tee -a "$LOGFILE"
fi

log ""
log "All done. Good luck with experimental validation!"