#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-bg-marco}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is required. Install Miniconda first." >&2
  exit 1
fi

conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# BoltzGen install (editable install from local repo for reproducibility)
pip install --upgrade pip setuptools wheel
pip install -e /workspace/boltzgen

# Optional helper packages for workflow scripts
pip install pandas biopython pyyaml numpy

echo "Environment '${ENV_NAME}' ready. Activate with: conda activate ${ENV_NAME}"
