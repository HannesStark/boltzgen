#!/usr/bin/env bash
set -euo pipefail

SPEC="${1:-specs/human_marco_nanobody_anywhere.yaml}"
OUTDIR="${2:-runs/nanobody_$(date +%Y%m%d_%H%M%S)}"
NUM_DESIGNS="${NUM_DESIGNS:-200}"
BUDGET="${BUDGET:-40}"
DEVICES="${DEVICES:-1}"

if [[ ! -f "$SPEC" ]]; then
  echo "ERROR: spec file not found: $SPEC" >&2
  echo "Usage: $0 [SPEC.yaml] [output_dir]" >&2
  exit 1
fi

# By default nanobody-anything avoids Cys in inverse folding.
# To permit Cys explicitly, pass e.g. EXTRA_ARGS='--inverse_fold_avoid ""'
EXTRA_ARGS="${EXTRA_ARGS:-}"

boltzgen run "$SPEC" \
  --output "$OUTDIR" \
  --protocol nanobody-anything \
  --num_designs "$NUM_DESIGNS" \
  --budget "$BUDGET" \
  --devices "$DEVICES" \
  --reuse \
  $EXTRA_ARGS
