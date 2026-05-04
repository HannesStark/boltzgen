#!/usr/bin/env bash
set -euo pipefail

SPEC="${1:-specs/crossreactive_conserved_surface.yaml}"
PROTOCOL="${2:-protein-anything}"
OUTDIR="${3:-runs/production_$(date +%Y%m%d_%H%M%S)}"
NUM_DESIGNS="${NUM_DESIGNS:-1500}"
BUDGET="${BUDGET:-200}"
DEVICES="${DEVICES:-1}"

boltzgen run "$SPEC" \
  --output "$OUTDIR" \
  --protocol "$PROTOCOL" \
  --num_designs "$NUM_DESIGNS" \
  --budget "$BUDGET" \
  --devices "$DEVICES"
