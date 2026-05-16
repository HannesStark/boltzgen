#!/usr/bin/env bash
# =============================================================================
# collect_campaign.sh — Gather and merge metrics from multiple BoltzGen campaigns
# =============================================================================
# Usage:
#   ./scripts/collect_campaign.sh \
#     --runs runs/mouse_vhh_prod runs/human_vhh_prod runs/cross_vhh_prod \
#     --out results/all_metrics.csv
#
# Adds `source_run` and `source_spec` columns to each row. Safe to re-run even
# if some campaigns are incomplete or failed.
# =============================================================================
set -euo pipefail

RUNS=()
OUT="results/all_metrics.csv"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runs) RUNS=("${@:2}"); shift $# ;;
    --out)  OUT="$2"; shift 2 ;;
    *)      echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ ${#RUNS[@]} -eq 0 ]]; then
  echo "Usage: $0 --runs RUN1 [RUN2 ...] --out OUTPUT.csv" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"

# Map run dirname → inferred spec name for labelling
declare -A RUN_SPEC_MAP=(
  ["mouse_vhh_prod"]="mouse_marco_nanobody_hotspot"
  ["human_vhh_prod"]="human_marco_nanobody_hotspot"
  ["cross_vhh_prod"]="crossreactive_marco_nanobody_hotspot"
)

PY_SCRIPT=$(mktemp)
TEMP_CSV=$(mktemp)

cat > "$PY_SCRIPT" << 'PYEOF'
import pandas as pd, sys, glob, os, tempfile, shutil

runs = []
specs = []
out_path = sys.argv[1] if len(sys.argv) > 1 else "results/all_metrics.csv"
run_dirs = sys.argv[2:]

merged_dfs = []

for run_dir in run_dirs:
    run_name = os.path.basename(run_dir.rstrip("/"))

    # Detect the metrics file — final_ranked_designs is populated after step 5
    metric_file = None
    for sub in ["all_designs_metrics.csv", "aggregate_metrics_analyze.csv"]:
        candidate = os.path.join(run_dir, "final_ranked_designs", sub)
        if os.path.isfile(candidate):
            metric_file = candidate
            break
        candidate2 = os.path.join(run_dir, sub)
        if os.path.isfile(candidate2):
            metric_file = candidate2
            break

    if not metric_file:
        print(f"WARNING: no metrics CSV in {run_dir} — skipping", file=sys.stderr)
        continue

    run_base = os.path.basename(run_dir)
    # Infer spec name from the metrics CSV filename or run directory
    # e.g. mouse_marco_nanobody_hotspot_057.csv → mouse_marco_nanobody_hotspot
    spec_name = run_base  # default: use the run directory name
    if os.path.basename(metric_file).startswith(run_base):
        stem = os.path.splitext(os.path.basename(metric_file))[0]
        # Strip trailing _number pattern
        import re
        stem_clean = re.sub(r'_\d+$', '', stem)
        if stem_clean:
            spec_name = stem_clean

    print(f"Collecting: {metric_file}  (spec={spec_name})")

    df = pd.read_csv(metric_file)
    df["source_run"] = run_name
    df["source_spec"] = spec_name
    df["metrics_file"] = metric_file
    merged_dfs.append(df)

if not merged_dfs:
    print("ERROR: no metrics files were found or readable", file=sys.stderr)
    sys.exit(1)

merged = pd.concat(merged_dfs, ignore_index=True)

# Deduplicate if the same design appears in multiple runs (keep best by pLDDT/ipTM)
if {"plddt", "ipTM", "ptm", "confidence"} & set(merged.columns):
    score_col = next((c for c in ["ipTM", "ptm", "plddt", "confidence"] if c in merged.columns), None)
    if score_col:
        merged = merged.sort_values(score_col, ascending=False).drop_duplicates(subset="design_id").sort_index()

merged.to_csv(out_path, index=False)
print(f"Wrote {out_path}  ({len(merged)} rows from {len(merged_dfs)} campaigns)")
PYEOF

python3 "$PY_SCRIPT" "$OUT" "${RUNS[@]}"
rm -f "$PY_SCRIPT" "$TEMP_CSV"