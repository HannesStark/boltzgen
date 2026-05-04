#!/usr/bin/env bash
set -euo pipefail

SPEC="${1:-specs/human_marco_nanobody_anywhere.yaml}"
OUTDIR="${2:-runs/nanobody_$(date +%Y%m%d_%H%M%S)}"
NUM_DESIGNS="${NUM_DESIGNS:-200}"
BUDGET="${BUDGET:-40}"
DEVICES="${DEVICES:-1}"

# By default nanobody-anything avoids Cys in inverse folding.
# To permit Cys explicitly, pass e.g. EXTRA_ARGS='--inverse_fold_avoid ""'
EXTRA_ARGS="${EXTRA_ARGS:-}"

boltzgen run "$SPEC" \
  --output "$OUTDIR" \
  --protocol nanobody-anything \
  --num_designs "$NUM_DESIGNS" \
  --budget "$BUDGET" \
  --devices "$DEVICES" \
  $EXTRA_ARGS
