#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — Standalone single-node pipeline for MARCO nanobody design.
# Works in two modes:
#
#   (A) HPC Singularity — recommended for GPU nodes:
#       export SIF=/path/to/ubuntu24_boltzgen.sif
#       bash scripts/run_pipeline.sh specs/...yaml run_name
#
#   (B) Local conda — for development/laptops:
#       conda activate boltzgen
#       bash scripts/run_pipeline.sh specs/...yaml run_name
#
# All 5 pipeline steps complete in one invocation.
#
# Environment variables (all optional):
#   SIF          Path to ubuntu24_boltzgen.sif  [for HPC singularity mode]
#   SPEC         Path to YAML spec file          [default: first arg]
#   OUTNAME      Run name for output dir         [default: second arg]
#   GPUS         GPU count                        (default: 4)
#   NUM_DESIGNS  Number of designs               (default: 60000)
#   BUDGET       Inference steps                 (default: 200)
#   SPEED_MODE   1=fast, 0=quality              (default: 0)
#   MIN_IPTM     ipTM threshold                  (default: 0.25)
#   TOP_N        Top designs to keep             (default: 500)
# =============================================================================

set -euo pipefail

# ── Input args ───────────────────────────────────────────────────────────────
SPEC="${1:-specs/crossreactive_marco_nanobody_setD_beta_pairing.yaml}"
OUTNAME="${2:-$(basename "$SPEC" .yaml)}"

# ── Config ───────────────────────────────────────────────────────────────────
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

# ── Thread limits ────────────────────────────────────────────────────────────
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ── Log helper ───────────────────────────────────────────────────────────────
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"
}

# ════════════════════════════════════════════════════════════════════════════
# MODE A: HPC — run inside Singularity/Apptainer container
# ════════════════════════════════════════════════════════════════════════════
if [[ -n "${SIF:-}" ]]; then

  if [[ ! -f "$SIF" ]]; then
    echo "ERROR: SIF not found: $SIF" >&2
    exit 1
  fi

  log "HPC mode — Singularity container: $SIF"

  # Write the pipeline body to a temp script that gets executed inside the SIF
  PIPELINE_SCRIPT="$(mktemp /tmp/pipeline_body_XXXXXX.sh)"
  trap "rm -f '$PIPELINE_SCRIPT'" EXIT

  # ── Pipeline body (runs inside the container) ────────────────────────────
  cat > "$PIPELINE_SCRIPT" << 'PIPELINE_EOF'
#!/usr/bin/env bash
set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

source /opt/venv/bin/activate

PROJECT_DIR="$(cat /proc/1/cwd 2>/dev/null || echo /project)"
cd "$PROJECT_DIR"

OUTDIR="$1"; LOGFILE="$2"; SPEC="$3"; GPUS="$4"; NUM_DESIGNS="$5"
BUDGET="$6"; SPEED_MODE="$7"; MIN_IPTM="$8"; TOP_N="$9"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"; }

SPEC_NAME="$(basename "$SPEC" .yaml)"
METRICS="$OUTDIR/final_ranked_designs/all_designs_metrics.csv"

# ══ STEP 1: BoltzGen generation ════════════════════════════════════════════
log "═══════════════════════════════════════════════════════════════════"
log "[Step 1] BoltzGen generation"
log "  spec=$SPEC outdir=$OUTDIR GPUS=$GPUS num_designs=$NUM_DESIGNS budget=$BUDGET speed=$SPEED_MODE"
log "═══════════════════════════════════════════════════════════════════"

if [[ "$SPEED_MODE" == "1" ]]; then
  FOLD_ARGS="--config fold sampling_steps=100 recycling_steps=1 diffusion_samples=1 compile_structure=true"
  DESIGN_ARGS="--config design compile_pairformer=true compile_structure=true"
  IFOLD_ARGS="--config inverse_fold precision=bf16-mixed"
else
  FOLD_ARGS="--config fold compile_structure=true"
  DESIGN_ARGS=""
  IFOLD_ARGS=""
fi

boltzgen run "$SPEC" \
  --output "$OUTDIR" \
  --protocol nanobody-anything \
  --num_designs $NUM_DESIGNS \
  --budget $BUDGET \
  --devices $GPUS \
  --reuse \
  --diffusion_batch_size 16 \
  --metrics_override plip_hbonds_refolded=0.2 delta_sasa_refolded=0.5 \
  --refolding_rmsd_threshold 3.0 \
  $FOLD_ARGS $DESIGN_ARGS $IFOLD_ARGS \
  2>&1 | tee -a "$LOGFILE"

if [[ ! -f "$METRICS" ]]; then
  log "[Step 1] ERROR: metrics not found at $METRICS"
  exit 1
fi

log "[Step 1] Done. Raw designs: $(wc -l < "$METRICS") lines"

# ══ STEP 2: N-glyc filter ══════════════════════════════════════════════════
log "═══════════════════════════════════════════════════════════════════"
log "[Step 2] N-glyc filter"
log "═══════════════════════════════════════════════════════════════════"
python scripts/filter_nglyc.py --metrics "$METRICS" --out "$METRICS" 2>&1 | tee -a "$LOGFILE"
log "[Step 2] Done. After filter: $(wc -l < "$METRICS") lines"

# ══ STEP 3: Developability filter ══════════════════════════════════════════
log "═══════════════════════════════════════════════════════════════════"
log "[Step 3] Developability filter"
log "═══════════════════════════════════════════════════════════════════"
python scripts/filter_developability.py \
  --metrics "$METRICS" --out "$METRICS" \
  --filter_nglyc --filter_proline --filter_mode hard \
  2>&1 | tee -a "$LOGFILE"
log "[Step 3] Done. After filter: $(wc -l < "$METRICS") lines"

# ══ STEP 4: Novelty check ══════════════════════════════════════════════════
log "═══════════════════════════════════════════════════════════════════"
log "[Step 4] CDR novelty check"
log "═══════════════════════════════════════════════════════════════════"
python scripts/novelty_check.py \
  --designs "$METRICS" --filter_mode both --min_edit_distance 4 --out "$METRICS" \
  2>&1 | tee -a "$LOGFILE"
log "[Step 4] Done. After filter: $(wc -l < "$METRICS") lines"

# ══ STEP 5: Rank ═══════════════════════════════════════════════════════════
log "═══════════════════════════════════════════════════════════════════"
log "[Step 5] Rank designs (min_iptm=$MIN_IPTM, top $TOP_N)"
log "═══════════════════════════════════════════════════════════════════"
TOP_CSV="$OUTDIR/top${TOP_N}_candidates.csv"
python scripts/rank_designs.py \
  --metrics "$METRICS" --min_iptm $MIN_IPTM --max_pae 15 \
  --max_gly_ala_frac 0.35 --top_n $TOP_N --out "$TOP_CSV" \
  2>&1 | tee -a "$LOGFILE"

log "═══════════════════════════════════════════════════════════════════"
log "PIPELINE COMPLETE"
log "  Full metrics : $METRICS"
log "  Top candidates: $TOP_CSV"
log "  Log file     : $LOGFILE"
log "═══════════════════════════════════════════════════════════════════"

if [[ -f "$TOP_CSV" ]]; then
  log ""
  log "─── Top 10 candidates ───"
  head -11 "$TOP_CSV" | awk -F',' 'NR==1{next} {printf "  %-6s ipTM=%.3f pAE=%.1f score=%.4f  %s\n", $1, $4, $5, $NF, $2}' | tee -a "$LOGFILE"
fi

log "All done."
PIPELINE_EOF

  chmod +x "$PIPELINE_SCRIPT"

  apptainer exec --nv \
    --bind "$PROJECT_DIR" \
    --bind "$(dirname "$SIF"):/opt/sif:ro" \
    "$SIF" \
    bash "$PIPELINE_SCRIPT" \
      "$OUTDIR" "$LOGFILE" "$SPEC" "$GPUS" "$NUM_DESIGNS" \
      "$BUDGET" "$SPEED_MODE" "$MIN_IPTM" "$TOP_N"

  exit $?
fi

# ════════════════════════════════════════════════════════════════════════════
# MODE B: Local — conda environment
# ════════════════════════════════════════════════════════════════════════════
log "Local mode — using conda"
eval "$(conda shell.bash hook)"
conda activate boltzgen

SPEC_NAME="$(basename "$SPEC" .yaml)"
METRICS="$OUTDIR/final_ranked_designs/all_designs_metrics.csv"

# ══ STEP 1: BoltzGen generation ════════════════════════════════════════════
log "═══════════════════════════════════════════════════════════════════"
log "[Step 1] BoltzGen generation"
log "  spec=$SPEC outdir=$OUTDIR GPUS=$GPUS num_designs=$NUM_DESIGNS budget=$BUDGET speed=$SPEED_MODE"
log "═══════════════════════════════════════════════════════════════════"

if [[ "$SPEED_MODE" == "1" ]]; then
  FOLD_ARGS="--config fold sampling_steps=100 recycling_steps=1 diffusion_samples=1 compile_structure=true"
  DESIGN_ARGS="--config design compile_pairformer=true compile_structure=true"
  IFOLD_ARGS="--config inverse_fold precision=bf16-mixed"
else
  FOLD_ARGS="--config fold compile_structure=true"
  DESIGN_ARGS=""
  IFOLD_ARGS=""
fi

boltzgen run "$SPEC" \
  --output "$OUTDIR" \
  --protocol nanobody-anything \
  --num_designs $NUM_DESIGNS \
  --budget $BUDGET \
  --devices $GPUS \
  --reuse \
  --diffusion_batch_size 16 \
  --metrics_override plip_hbonds_refolded=0.2 delta_sasa_refolded=0.5 \
  --refolding_rmsd_threshold 3.0 \
  $FOLD_ARGS $DESIGN_ARGS $IFOLD_ARGS \
  2>&1 | tee -a "$LOGFILE"

if [[ ! -f "$METRICS" ]]; then
  log "[Step 1] ERROR: metrics not found at $METRICS"
  exit 1
fi

log "[Step 1] Done. Raw designs: $(wc -l < "$METRICS") lines"

# ══ STEP 2: N-glyc filter ══════════════════════════════════════════════════
log "═══════════════════════════════════════════════════════════════════"
log "[Step 2] N-glyc filter"
log "═══════════════════════════════════════════════════════════════════"
python scripts/filter_nglyc.py --metrics "$METRICS" --out "$METRICS" 2>&1 | tee -a "$LOGFILE"
log "[Step 2] Done. After filter: $(wc -l < "$METRICS") lines"

# ══ STEP 3: Developability filter ══════════════════════════════════════════
log "═══════════════════════════════════════════════════════════════════"
log "[Step 3] Developability filter"
log "═══════════════════════════════════════════════════════════════════"
python scripts/filter_developability.py \
  --metrics "$METRICS" --out "$METRICS" \
  --filter_nglyc --filter_proline --filter_mode hard \
  2>&1 | tee -a "$LOGFILE"
log "[Step 3] Done. After filter: $(wc -l < "$METRICS") lines"

# ══ STEP 4: Novelty check ══════════════════════════════════════════════════
log "═══════════════════════════════════════════════════════════════════"
log "[Step 4] CDR novelty check"
log "═══════════════════════════════════════════════════════════════════"
python scripts/novelty_check.py \
  --designs "$METRICS" --filter_mode both --min_edit_distance 4 --out "$METRICS" \
  2>&1 | tee -a "$LOGFILE"
log "[Step 4] Done. After filter: $(wc -l < "$METRICS") lines"

# ══ STEP 5: Rank ═══════════════════════════════════════════════════════════
log "═══════════════════════════════════════════════════════════════════"
log "[Step 5] Rank designs (min_iptm=$MIN_IPTM, top $TOP_N)"
log "═══════════════════════════════════════════════════════════════════"
TOP_CSV="$OUTDIR/top${TOP_N}_candidates.csv"
python scripts/rank_designs.py \
  --metrics "$METRICS" --min_iptm $MIN_IPTM --max_pae 15 \
  --max_gly_ala_frac 0.35 --top_n $TOP_N --out "$TOP_CSV" \
  2>&1 | tee -a "$LOGFILE"

log "═══════════════════════════════════════════════════════════════════"
log "PIPELINE COMPLETE"
log "  Full metrics : $METRICS"
log "  Top candidates: $TOP_CSV"
log "  Log file     : $LOGFILE"
log "═══════════════════════════════════════════════════════════════════"

if [[ -f "$TOP_CSV" ]]; then
  log ""
  log "─── Top 10 candidates ───"
  head -11 "$TOP_CSV" | awk -F',' 'NR==1{next} {printf "  %-6s ipTM=%.3f pAE=%.1f score=%.4f  %s\n", $1, $4, $5, $NF, $2}' | tee -a "$LOGFILE"
fi

log "All done."