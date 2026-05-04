#!/usr/bin/env bash
set -euo pipefail

SPEC="${1:-specs/human_marco_binder_anywhere.yaml}"
PROTOCOL="${2:-protein-anything}"
OUTDIR="${3:-runs/pilot_$(date +%Y%m%d_%H%M%S)}"
NUM_DESIGNS="${NUM_DESIGNS:-50}"
BUDGET="${BUDGET:-10}"
DEVICES="${DEVICES:-1}"

boltzgen run "$SPEC" \
  --output "$OUTDIR" \
  --protocol "$PROTOCOL" \
  --num_designs "$NUM_DESIGNS" \
  --budget "$BUDGET" \
  --devices "$DEVICES"
