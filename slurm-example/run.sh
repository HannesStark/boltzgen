#!/bin/bash
set -e
set -x

DESIGN_SPEC=/path/to/design/spec.yaml
MERGED_OUT=/path/to/your/output/directory
NUM_TASKS=20
NUM_DESIGNS_PER_TASK=1000
CONDA_ENVIRONMENT=/path/to/conda/environment/with/boltzgen
CACHE_DIR=/path/where/models/are/saved
ACCOUNT=FILL_THIS_IN
TIME=05:00:00

OUT="${MERGED_OUT}/task-outputs"
LOGS="${MERGED_OUT}/task-logs"

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 {submit|process}"
    exit 1
fi


MODE="$1"

if [[ "$MODE" == "submit" ]]; then

    mkdir -p "$OUT"
    mkdir -p "$LOGS"

    sbatch -A "$ACCOUNT" -t "$TIME" --export=ALL --array=1-$NUM_TASKS -o $LOGS/stdout.%A-%a.log -e $LOGS/stderr.%A-%a.log run_job_array.slurm \
        "$DESIGN_SPEC" "$OUT" "$NUM_DESIGNS_PER_TASK" "$CONDA_ENVIRONMENT" --protocol protein-anything --cache "$CACHE_DIR"
    squeue --me

elif [[ "$MODE" == "process" ]]; then

    boltzgen merge "$OUT"/task-* --output "$MERGED_OUT"
    boltzgen run "$DESIGN_SPEC" --steps filtering --protocol protein-anything --output "$MERGED_OUT"

else
    echo "Usage: $0 {submit|process}"
    exit 1
fi
