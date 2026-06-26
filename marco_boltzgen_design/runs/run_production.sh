#!/usr/bin/env bash
set -euo pipefail

# ── Thread limits ────────────────────────────────────────────────────────────
# OpenBLAS, OpenMP, MKL each try to spawn many threads by default.
# On shared HPC nodes this can exhaust RLIMIT_NPROC and cause:
#   "pthread_create failed: Resource temporarily unavailable"
# Cap them to 1 thread each — all heavy GPU work is done by BoltzGen
# via PyTorch, which manages its own thread pool independently.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

SPEC="${1:-specs/crossreactive_marco_nanobody_setC_hybrid.yaml}"
PROTOCOL="${2:-nanobody-anything}"
OUTDIR="${3:-runs/production_$(date +%Y%m%d_%H%M%S)}"
NUM_DESIGNS="${NUM_DESIGNS:-1500}"
BUDGET="${BUDGET:-200}"
DEVICES="${DEVICES:-1}"
DIFFUSION_ARGS="${DIFFUSION_ARGS:-}"
MARCO_EXTRA_ARGS="${MARCO_EXTRA_ARGS:---diffusion_batch_size 2 --metrics_override plip_hbonds_refolded=0.2 delta_sasa_refolded=0.5 --refolding_rmsd_threshold 3.0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

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
  $MARCO_EXTRA_ARGS \
  $DIFFUSION_ARGS \
  $EXTRA_ARGS
