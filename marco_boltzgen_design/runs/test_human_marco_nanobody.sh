#!/usr/bin/env bash
set -euo pipefail

# Quick pilot test for human MARCO nanobody design specs.
# Usage:
#   ./runs/test_human_marco_nanobody.sh
# Optional overrides:
#   NUM_DESIGNS=20 BUDGET=5 DEVICES=1 ./runs/test_human_marco_nanobody.sh

cd "$(dirname "$0")/.."

export NUM_DESIGNS="${NUM_DESIGNS:-10}"
export BUDGET="${BUDGET:-3}"
export DEVICES="${DEVICES:-1}"
# NOTE: diffusion params (num_steps, guidance_scale) are SPEC-ONLY settings.
# They CANNOT be passed as boltzgen CLI args. To override them, edit the
# spec YAML directly or set diffusion values in the spec before running.
# Any CLI EXTRA_ARGS must be valid boltzgen run flags (see: boltzgen run --help).
export EXTRA_ARGS="${EXTRA_ARGS:-}"

STAMP="$(date +%Y%m%d_%H%M%S)"

echo "[1/4] Checking required target/spec files"
test -f "targets/human_MARCO_input.cif"
test -f "specs/human_marco_nanobody_anywhere.yaml"
test -f "specs/human_marco_nanobody_hotspot.yaml"

echo "[2/4] Validating YAML specs with boltzgen check"
boltzgen check specs/human_marco_nanobody_anywhere.yaml
boltzgen check specs/human_marco_nanobody_hotspot.yaml

echo "[3/4] Running ANYWHERE pilot"
./runs/run_nanobody_campaign.sh \
  specs/human_marco_nanobody_anywhere.yaml \
  "runs/test_human_nanobody_anywhere_${STAMP}"

echo "[4/4] Running HOTSPOT pilot"
./runs/run_nanobody_campaign.sh \
  specs/human_marco_nanobody_hotspot.yaml \
  "runs/test_human_nanobody_hotspot_${STAMP}"

echo "Done. Outputs:"
echo "  runs/test_human_nanobody_anywhere_${STAMP}"
echo "  runs/test_human_nanobody_hotspot_${STAMP}"
