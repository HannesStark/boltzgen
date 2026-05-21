#!/usr/bin/env bash
set -euo pipefail

SPEC="${1:-specs/human_marco_nanobody_anywhere.yaml}"
OUTDIR="${2:-runs/nanobody_$(date +%Y%m%d_%H%M%S)}"
NUM_DESIGNS="${NUM_DESIGNS:-200}"
BUDGET="${BUDGET:-40}"
# BoltzGen manual: --devices defaults to all visible GPUs.
# For dual-GPU MARCO nodes, default to 2 explicitly for reproducibility.
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

# MARCO SRCR defaults tuned for interface quality on limited VRAM.
MARCO_EXTRA_ARGS="${MARCO_EXTRA_ARGS:---diffusion_batch_size 2 --metrics_override plip_hbonds_refolded=0.2 delta_sasa_refolded=0.5 --refolding_rmsd_threshold 3.0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$OUTDIR"

echo "[marco-run] spec=$SPEC"
echo "[marco-run] outdir=$OUTDIR"
echo "[marco-run] protocol=$PROTOCOL"
echo "[marco-run] num_designs=$NUM_DESIGNS budget=$BUDGET devices=$DEVICES"

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
