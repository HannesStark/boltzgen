#!/usr/bin/env bash
set -euo pipefail

# Quick pilot test for human MARCO nanobody design specs.
# Usage:
#   ./runs/test_human_marco_nanobody.sh
# Optional overrides:
#   NUM_DESIGNS=20 BUDGET=5 DEVICES=1 EXTRA_ARGS='--diffusion.num_steps 200 --diffusion.guidance_scale 0.2' ./runs/test_human_marco_nanobody.sh

cd "$(dirname "$0")/.."

export NUM_DESIGNS="${NUM_DESIGNS:-10}"
export BUDGET="${BUDGET:-3}"
export DEVICES="${DEVICES:-1}"
export EXTRA_ARGS="${EXTRA_ARGS:---diffusion.num_steps 120 --diffusion.guidance_scale 0.2}"

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
