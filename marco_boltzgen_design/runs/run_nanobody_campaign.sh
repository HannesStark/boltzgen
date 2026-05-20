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

# Determine protocol from spec comment (first match), defaulting to nanobody-anything.
# The spec YAML file may contain a comment like "# Protocol: nanobody-hotspot"
# which is more appropriate for hotspot-constrained designs than forcing
# nanobody-anything, which ignores binding-type constraints.
PROTOCOL_FROM_SPEC="$(grep -m1 -oP '(?<=Protocol:\s)\w+(?:-\w+)*' "$SPEC" 2>/dev/null || true)"
PROTOCOL="${PROTOCOL:-$PROTOCOL_FROM_SPEC}"
PROTOCOL="${PROTOCOL:-nanobody-anything}"

# MARCO SRCR is polar/beta-rich. Keep batches small for length diversity,
# up-weight buried H-bonds/SASA in filtering, and relax VHH-loop refolding RMSD.
MARCO_EXTRA_ARGS="${MARCO_EXTRA_ARGS:---diffusion_batch_size 2 --metrics_override plip_hbonds_refolded=0.2 delta_sasa_refolded=0.5 --refolding_rmsd_threshold 3.0}"

# By default nanobody-anything avoids Cys in inverse folding.
# To permit Cys explicitly, append e.g. EXTRA_ARGS='--inverse_fold_avoid ""'
EXTRA_ARGS="${EXTRA_ARGS:-}"

boltzgen run "$SPEC" \
  --output "$OUTDIR" \
  --protocol "$PROTOCOL" \
  --num_designs "$NUM_DESIGNS" \
  --budget "$BUDGET" \
  --devices "$DEVICES" \
  --reuse \
  $MARCO_EXTRA_ARGS \
  $EXTRA_ARGS
