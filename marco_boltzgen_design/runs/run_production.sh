#!/usr/bin/env bash
set -euo pipefail

SPEC="${1:-specs/crossreactive_conserved_surface.yaml}"
PROTOCOL="${2:-protein-anything}"
OUTDIR="${3:-runs/production_$(date +%Y%m%d_%H%M%S)}"
NUM_DESIGNS="${NUM_DESIGNS:-1500}"
BUDGET="${BUDGET:-200}"
DEVICES="${DEVICES:-1}"
DIFFUSION_ARGS="${DIFFUSION_ARGS:-}"

if [[ ! -f "$SPEC" ]]; then
  echo "ERROR: spec file not found: $SPEC" >&2
  echo "Usage: $0 [SPEC.yaml] [protocol] [output_dir]" >&2
  exit 1
fi

boltzgen run "$SPEC" \
  --output "$OUTDIR" \
  --protocol "$PROTOCOL" \
  --num_designs "$NUM_DESIGNS" \
  --budget "$BUDGET" \
  --devices "$DEVICES" \
  --reuse \
  $DIFFUSION_ARGS
