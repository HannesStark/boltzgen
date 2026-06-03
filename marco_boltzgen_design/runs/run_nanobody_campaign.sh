#!/usr/bin/env bash
set -euo pipefail

SPEC="${1:-specs/human_marco_nanobody_anywhere.yaml}"
OUTDIR="${2:-runs/nanobody_$(date +%Y%m%d_%H%M%S)}"
NUM_DESIGNS="${NUM_DESIGNS:-200}"
BUDGET="${BUDGET:-40}"
DEVICES="${DEVICES:-2}"

if [[ ! -f "$SPEC" ]]; then
  echo "ERROR: spec file not found: $SPEC" >&2
  echo "Usage: $0 [SPEC.yaml] [output_dir]" >&2
  exit 1
fi

# Determine protocol from a metadata comment in spec, e.g.:
#   # Protocol: nanobody-hotspot
# If absent, fallback to nanobody-anything.
PROTOCOL_FROM_SPEC="$(awk 'match($0, /Protocol:[[:space:]]*([A-Za-z0-9-]+)/, m) {print m[1]; exit}' "$SPEC" 2>/dev/null || true)"
PROTOCOL="${PROTOCOL:-$PROTOCOL_FROM_SPEC}"
PROTOCOL="${PROTOCOL:-nanobody-anything}"

# ── Speed mode ──────────────────────────────────────────────────────────────
# SPEED_MODE=1  →  aggressive speedup (screening / large batches)
# SPEED_MODE=0  →  default balanced quality (or unset)
SPEED_MODE="${SPEED_MODE:-0}"

if [[ "$SPEED_MODE" == "1" ]]; then
  echo "[marco-run] SPEED_MODE=1 — using fast folding config (sampling_steps=100, recycling_steps=1, diffusion_samples=1)"
fi

# ── MARCO defaults ───────────────────────────────────────────────────────────
# diffusion_batch_size: RTX 5000 (16 GB) handles more than the old default of 2.
#   Speed mode bumps it to 8 for better GPU utilization.
DEFAULT_DIFFUSION_BATCH_SIZE=2
[[ "$SPEED_MODE" == "1" ]] && DEFAULT_DIFFUSION_BATCH_SIZE=8

MARCO_EXTRA_ARGS="${MARCO_EXTRA_ARGS:---diffusion_batch_size $DEFAULT_DIFFUSION_BATCH_SIZE --metrics_override plip_hbonds_refolded=0.2 delta_sasa_refolded=0.5 --refolding_rmsd_threshold 3.0}"

# ── Speed-mode fold/inverse_fold overrides ──────────────────────────────────
# folding:    sampling_steps 200→100, recycling_steps 3→1, diffusion_samples 5→1
# inverse_fold: precision FP32→bf16-mixed
# design:     compile_pairformer + compile_structure for ~20-40% speedup
SPEED_FOLD_ARGS="--config fold sampling_steps=100 recycling_steps=1 diffusion_samples=1"
SPEED_DESIGN_ARGS="--config design compile_pairformer=true compile_structure=true"
SPEED_IFOLD_ARGS="--config inverse_fold precision=bf16-mixed"

EXTRA_ARGS="${EXTRA_ARGS:-}"
if [[ "$SPEED_MODE" == "1" ]]; then
  EXTRA_ARGS="$SPEED_FOLD_ARGS $SPEED_DESIGN_ARGS $SPEED_IFOLD_ARGS $EXTRA_ARGS"
fi

mkdir -p "$OUTDIR"

echo "[marco-run] spec=$SPEC"
echo "[marco-run] outdir=$OUTDIR"
echo "[marco-run] protocol=$PROTOCOL"
echo "[marco-run] num_designs=$NUM_DESIGNS budget=$BUDGET devices=$DEVICES speed_mode=$SPEED_MODE"

cmd=(
  boltzgen run "$SPEC"
  --output "$OUTDIR"
  --protocol "$PROTOCOL"
  --num_designs "$NUM_DESIGNS"
  --budget "$BUDGET"
  --devices "$DEVICES"
  --reuse
)

# shellcheck disable=SC2206
marco_extra_arr=( $MARCO_EXTRA_ARGS )
# shellcheck disable=SC2206
extra_arr=( $EXTRA_ARGS )

cmd+=("${marco_extra_arr[@]}" "${extra_arr[@]}")

"${cmd[@]}"